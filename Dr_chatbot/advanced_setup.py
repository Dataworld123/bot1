"""
Advanced Setup and Run Script for Dr. Meenakshi Tomar Dental Chatbot
Initializes all advanced AI components and starts the enhanced server
"""

import os
import sys
import asyncio
import uvicorn
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import openai
from pinecone import Pinecone

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import our modules
from check_chunks import check_data_chunks
from final_load import load_data_to_pinecone
from advanced_app import app

def setup_logging():
    """Configure advanced logging"""
    
    # Remove default logger
    logger.remove()
    
    # Add console logger with colors
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True
    )
    
    # Add file logger
    log_dir = current_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logger.add(
        log_dir / "advanced_chatbot.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="1 day",
        retention="7 days",
        compression="zip"
    )
    
    logger.info("🚀 Advanced logging system initialized")

def check_environment():
    """Check all required environment variables and dependencies"""
    
    logger.info("🔍 Checking environment configuration...")
    
    # Load environment variables
    load_dotenv()
    
    required_vars = [
        'OPENAI_API_KEY',
        'PINECONE_API_KEY',
        'PINECONE_ENVIRONMENT'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please check your .env file")
        return False
    
    # Test OpenAI connection
    try:
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        logger.info("✅ OpenAI API connection successful")
    except Exception as e:
        logger.error(f"❌ OpenAI API connection failed: {e}")
        return False
    
    # Test Pinecone connection
    try:
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        indexes = pc.list_indexes()
        logger.info("✅ Pinecone API connection successful")
    except Exception as e:
        logger.error(f"❌ Pinecone API connection failed: {e}")
        return False
    
    logger.info("✅ Environment configuration check complete")
    return True

def setup_data():
    """Setup and validate data for the advanced system"""
    
    logger.info("📊 Setting up data for advanced AI system...")
    
    # Check if data file exists
    data_file = current_dir / "plain_text_crawled_data (1) (1).txt"
    if not data_file.exists():
        logger.error(f"❌ Data file not found: {data_file}")
        return False
    
    # Check data chunks
    try:
        logger.info("🔍 Validating data chunks...")
        word_count, chunk_count = check_data_chunks()
        if word_count and chunk_count:
            logger.info(f"✅ Data validation complete: {word_count:,} words, {chunk_count} chunks")

            # Only load data if we have very few chunks
            if chunk_count < 100:
                logger.info("📤 Loading data to Pinecone vector database...")
                vector_count = load_data_to_pinecone()
                logger.info(f"✅ Data loaded to Pinecone: {vector_count} vectors")
            else:
                logger.info(f"✅ Data already loaded in Pinecone: {chunk_count} vectors")
        else:
            logger.error("❌ Could not validate data chunks")
            return False
    except Exception as e:
        logger.error(f"❌ Data setup failed: {e}")
        return False
    
    return True

def check_advanced_dependencies():
    """Check if all advanced AI dependencies are available"""
    
    logger.info("🔧 Checking advanced AI dependencies...")
    
    required_packages = [
        'openai',
        'pinecone',
        'fastapi',
        'uvicorn',
        'tiktoken',
        'loguru'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.debug(f"✅ {package} available")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"⚠️ {package} not available")
    
    if missing_packages:
        logger.error(f"❌ Missing packages: {', '.join(missing_packages)}")
        logger.error("Please install missing packages: pip install -r requirements.txt")
        return False
    
    logger.info("✅ All advanced AI dependencies available")
    return True

def create_directories():
    """Create necessary directories for the advanced system"""
    
    logger.info("📁 Creating necessary directories...")
    
    directories = [
        "logs",
        "conversation_memory",
        "temp"
    ]
    
    for directory in directories:
        dir_path = current_dir / directory
        dir_path.mkdir(exist_ok=True)
        logger.debug(f"✅ Directory created/verified: {directory}")
    
    logger.info("✅ Directory structure ready")

def display_system_info():
    """Display advanced system information"""
    
    logger.info("🎯 Advanced AI Dental Chatbot System Information:")
    logger.info("=" * 60)
    logger.info("🤖 AI Features:")
    logger.info("   • Multi-Agent System (Diagnostic, Treatment, Prevention, Emergency)")
    logger.info("   • Chain-of-Thought Reasoning")
    logger.info("   • Automatic Response Quality Checking")
    logger.info("   • Intelligent Reprompting for Better Responses")
    logger.info("   • Advanced RAG with OpenAI Embeddings")
    logger.info("   • Conversation Memory and Context Management")
    logger.info("   • Query Expansion and Relevance Boosting")
    logger.info("")
    logger.info("🏥 Medical Expertise:")
    logger.info("   • Dr. Meenakshi Tomar's 30+ years experience")
    logger.info("   • Specialized in full mouth reconstruction")
    logger.info("   • WCLI certified laser surgery")
    logger.info("   • Evidence-based treatment recommendations")
    logger.info("")
    logger.info("🔧 Technical Stack:")
    logger.info("   • OpenAI GPT-4o-mini with advanced prompting")
    logger.info("   • OpenAI text-embedding-3-small for embeddings")
    logger.info("   • Pinecone vector database for knowledge retrieval")
    logger.info("   • FastAPI with async processing")
    logger.info("   • Advanced query expansion and relevance scoring")
    logger.info("=" * 60)

async def start_advanced_server():
    """Start the advanced AI server"""
    
    logger.info("🚀 Starting Advanced AI Dental Chatbot Server...")
    
    config = uvicorn.Config(
        app=app,
        host="127.0.0.1",
        port=8000,
        log_level="info",
        reload=False,  # Disable reload for production
        access_log=True
    )
    
    server = uvicorn.Server(config)
    
    logger.info("🌐 Server starting on http://127.0.0.1:8000")
    logger.info("📱 Advanced chat interface available at: http://127.0.0.1:8000/advanced_chat.html")
    logger.info("🔍 Health check endpoint: http://127.0.0.1:8000/health")
    logger.info("📊 System stats endpoint: http://127.0.0.1:8000/stats")
    
    await server.serve()

def main():
    """Main setup and run function"""
    
    print("🦷 Dr. Meenakshi Tomar - Advanced AI Dental Chatbot")
    print("=" * 60)
    
    # Setup logging first
    setup_logging()
    
    # Create necessary directories
    create_directories()
    
    # Check dependencies
    if not check_advanced_dependencies():
        logger.error("❌ Dependency check failed. Please install required packages.")
        sys.exit(1)
    
    # Check environment
    if not check_environment():
        logger.error("❌ Environment check failed. Please check your configuration.")
        sys.exit(1)
    
    # Setup data
    if not setup_data():
        logger.error("❌ Data setup failed. Please check your data files.")
        sys.exit(1)
    
    # Display system information
    display_system_info()
    
    # Start the advanced server
    try:
        asyncio.run(start_advanced_server())
    except KeyboardInterrupt:
        logger.info("👋 Server shutdown requested by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
