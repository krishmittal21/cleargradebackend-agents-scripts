"""
Real-time voice handler for David agent.
Handles WebSocket connections for bidirectional voice streaming.
Uses Google Cloud Speech-to-Text and Text-to-Speech for audio processing.
"""

import asyncio
import base64
import json
import logging
from typing import Callable, Optional
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

# Audio configuration
SAMPLE_RATE_INPUT = 16000  # 16kHz for input audio
SAMPLE_RATE_OUTPUT = 24000  # 24kHz for output audio
AUDIO_ENCODING = "LINEAR16"  # PCM audio format


class VoiceSession:
    """Manages a single voice conversation session."""
    
    def __init__(
        self,
        websocket: WebSocket,
        session_id: str,
        user_id: str,
        agent_builder: Callable
    ):
        self.websocket = websocket
        self.session_id = session_id
        self.user_id = user_id
        self.agent_builder = agent_builder
        self.agent = None
        self.is_active = True
        self.audio_buffer = bytearray()
        self.speech_client = None
        self.tts_client = None
        
    async def initialize(self):
        """Initialize the agent and speech clients."""
        try:
            # Import Google Cloud clients
            from google.cloud import speech_v1 as speech
            from google.cloud import texttospeech_v1 as texttospeech
            
            self.speech_client = speech.SpeechAsyncClient()
            self.tts_client = texttospeech.TextToSpeechAsyncClient()
            
            # Build the David agent
            self.agent = self.agent_builder(self.session_id, self.user_id)
            
            logger.info(f"Voice session initialized: {self.session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize voice session: {e}")
            return False
    
    async def transcribe_audio(self, audio_data: bytes) -> Optional[str]:
        """Convert audio to text using Google Speech-to-Text."""
        try:
            from google.cloud import speech_v1 as speech
            
            audio = speech.RecognitionAudio(content=audio_data)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=SAMPLE_RATE_INPUT,
                language_code="en-US",
                enable_automatic_punctuation=True,
            )
            
            response = await self.speech_client.recognize(
                config=config,
                audio=audio
            )
            
            if response.results:
                transcript = response.results[0].alternatives[0].transcript
                logger.info(f"Transcribed: {transcript}")
                return transcript
            return None
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None
    
    async def generate_speech(self, text: str) -> Optional[bytes]:
        """Convert text to speech using Google Text-to-Speech."""
        try:
            from google.cloud import texttospeech_v1 as texttospeech
            
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name="en-US-Neural2-D",  # Male neural voice
                ssml_gender=texttospeech.SsmlVoiceGender.MALE
            )
            
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                sample_rate_hertz=SAMPLE_RATE_OUTPUT,
            )
            
            response = await self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            logger.info(f"Generated speech: {len(response.audio_content)} bytes")
            return response.audio_content
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None
    
    async def process_message(self, text: str) -> str:
        """Send message to David agent and get response."""
        try:
            response = await self.agent.ainvoke({"input": text})
            output = response.get("output", "I couldn't process that request.")
            logger.info(f"Agent response: {output[:100]}...")
            return output
        except Exception as e:
            logger.error(f"Agent error: {e}")
            return "Sorry, I encountered an error processing your request."
    
    async def send_event(self, event_type: str, data: dict):
        """Send an event to the client."""
        try:
            await self.websocket.send_json({
                "type": event_type,
                **data
            })
        except Exception as e:
            logger.error(f"Send event error: {e}")
    
    async def handle_audio_chunk(self, audio_base64: str):
        """Handle incoming audio chunk from client."""
        try:
            # Decode base64 audio and add to buffer
            audio_data = base64.b64decode(audio_base64)
            self.audio_buffer.extend(audio_data)
        except Exception as e:
            logger.error(f"Audio chunk error: {e}")
    
    async def process_audio_buffer(self):
        """Process accumulated audio buffer."""
        if not self.audio_buffer:
            return
        
        audio_data = bytes(self.audio_buffer)
        self.audio_buffer.clear()
        
        # Notify client we're processing
        await self.send_event("status", {"state": "transcribing"})
        
        # Transcribe the audio
        transcript = await self.transcribe_audio(audio_data)
        
        if not transcript:
            await self.send_event("error", {"message": "Could not transcribe audio"})
            return
        
        # Send transcript to client
        await self.send_event("transcript", {"text": transcript})
        
        # Notify client we're thinking
        await self.send_event("status", {"state": "thinking"})
        
        # Get response from David
        response_text = await self.process_message(transcript)
        
        # Send text response
        await self.send_event("response", {"text": response_text})
        
        # Notify client we're generating speech
        await self.send_event("status", {"state": "speaking"})
        
        # Generate speech
        audio_response = await self.generate_speech(response_text)
        
        if audio_response:
            # Send audio in chunks to prevent memory issues
            chunk_size = 32000  # ~1 second of audio at 16kHz
            for i in range(0, len(audio_response), chunk_size):
                chunk = audio_response[i:i + chunk_size]
                audio_base64 = base64.b64encode(chunk).decode('utf-8')
                await self.send_event("audio", {
                    "data": audio_base64,
                    "sampleRate": SAMPLE_RATE_OUTPUT,
                    "isLast": i + chunk_size >= len(audio_response)
                })
        
        # Notify client we're done
        await self.send_event("status", {"state": "idle"})
    
    async def close(self):
        """Clean up session resources."""
        self.is_active = False
        logger.info(f"Voice session closed: {self.session_id}")


async def websocket_voice_handler(
    websocket: WebSocket,
    session_id: str,
    user_id: str,
    agent_builder: Callable
):
    """
    Handle WebSocket connection for real-time voice conversation.
    
    Protocol:
    - Client sends: {"type": "audio", "data": "<base64 PCM audio>"}
    - Client sends: {"type": "end_audio"} when done speaking
    - Server sends: {"type": "status", "state": "transcribing|thinking|speaking|idle"}
    - Server sends: {"type": "transcript", "text": "<user's speech>"}
    - Server sends: {"type": "response", "text": "<David's response>"}
    - Server sends: {"type": "audio", "data": "<base64 PCM audio>", "sampleRate": 24000, "isLast": bool}
    - Server sends: {"type": "error", "message": "<error details>"}
    """
    
    session = VoiceSession(websocket, session_id, user_id, agent_builder)
    
    # Initialize the session
    if not await session.initialize():
        await session.send_event("error", {"message": "Failed to initialize voice session"})
        return
    
    await session.send_event("status", {"state": "ready"})
    
    try:
        while session.is_active:
            # Receive message from client
            data = await websocket.receive_json()
            message_type = data.get("type")
            
            if message_type == "audio":
                # Accumulate audio chunks
                await session.handle_audio_chunk(data.get("data", ""))
                
            elif message_type == "end_audio":
                # Process the accumulated audio
                await session.process_audio_buffer()
                
            elif message_type == "text":
                # Direct text input (for testing or fallback)
                text = data.get("text", "")
                if text:
                    await session.send_event("status", {"state": "thinking"})
                    response_text = await session.process_message(text)
                    await session.send_event("response", {"text": response_text})
                    
                    # Generate and send speech
                    await session.send_event("status", {"state": "speaking"})
                    audio_response = await session.generate_speech(response_text)
                    
                    if audio_response:
                        chunk_size = 32000
                        for i in range(0, len(audio_response), chunk_size):
                            chunk = audio_response[i:i + chunk_size]
                            audio_base64 = base64.b64encode(chunk).decode('utf-8')
                            await session.send_event("audio", {
                                "data": audio_base64,
                                "sampleRate": SAMPLE_RATE_OUTPUT,
                                "isLast": i + chunk_size >= len(audio_response)
                            })
                    
                    await session.send_event("status", {"state": "idle"})
                
            elif message_type == "ping":
                # Keep-alive ping
                await session.send_event("pong", {})
                
            elif message_type == "close":
                # Client requested close
                break
                
    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {session_id}")
    except Exception as e:
        logger.error(f"Voice handler error: {e}")
        try:
            await session.send_event("error", {"message": str(e)})
        except:
            pass
    finally:
        await session.close()
