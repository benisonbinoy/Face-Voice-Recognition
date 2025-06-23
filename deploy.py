"""
Deployment and Production Setup Script
Use this for setting up the system in production or different environments
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

class DeploymentManager:
    def __init__(self):
        self.project_root = Path.cwd()
        self.required_files = [
            'app.py', 'requirements.txt', 'config.py',
            'templates/index.html', 'README.md'
        ]
        self.required_dirs = ['face_data', 'voice_data', 'templates']
    
    def check_environment(self):
        """Check if the current environment is properly set up"""
        print("🔍 Checking environment...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            print("❌ Python 3.8+ required")
            return False
        
        print(f"✅ Python {sys.version}")
        
        # Check required files
        missing_files = []
        for file_path in self.required_files:
            if not (self.project_root / file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            print(f"❌ Missing files: {missing_files}")
            return False
        
        print("✅ All required files present")
        
        # Check required directories
        missing_dirs = []
        for dir_path in self.required_dirs:
            if not (self.project_root / dir_path).exists():
                missing_dirs.append(dir_path)
        
        if missing_dirs:
            print(f"❌ Missing directories: {missing_dirs}")
            return False
        
        print("✅ All required directories present")
        return True
    
    def create_production_config(self):
        """Create production configuration"""
        print("⚙️  Creating production configuration...")
        
        prod_config = '''# Production Configuration
import os

# Override development settings for production
SERVER_CONFIG = {
    'host': '0.0.0.0',
    'port': int(os.environ.get('PORT', 5000)),
    'debug': False
}

# Security settings
SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# Performance settings
MODEL_CACHE_SIZE = 100
MAX_UPLOAD_SIZE = 16 * 1024 * 1024  # 16MB

# Logging
import logging
logging.basicConfig(level=logging.INFO)
'''
        
        with open('prod_config.py', 'w') as f:
            f.write(prod_config)
        
        print("✅ Production config created")
    
    def create_docker_files(self):
        """Create Docker configuration for containerized deployment"""
        print("🐳 Creating Docker files...")
        
        dockerfile = '''FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libopencv-dev \\
    python3-opencv \\
    libsndfile1 \\
    portaudio19-dev \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for models and data
RUN mkdir -p face_data voice_data models

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Run the application
CMD ["python", "app.py"]
'''
        
        docker_compose = '''version: '3.8'

services:
  face-voice-recognition:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./face_data:/app/face_data
      - ./voice_data:/app/voice_data
      - ./models:/app/models
    environment:
      - FLASK_ENV=production
      - SECRET_KEY=your-production-secret-key
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - face-voice-recognition
    restart: unless-stopped
'''
        
        nginx_conf = '''events {
    worker_connections 1024;
}

http {
    upstream app {
        server face-voice-recognition:5000;
    }
    
    server {
        listen 80;
        client_max_body_size 50M;
        
        location / {
            proxy_pass http://app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
'''
        
        with open('Dockerfile', 'w') as f:
            f.write(dockerfile)
        
        with open('docker-compose.yml', 'w') as f:
            f.write(docker_compose)
        
        with open('nginx.conf', 'w') as f:
            f.write(nginx_conf)
        
        print("✅ Docker files created")
    
    def create_systemd_service(self):
        """Create systemd service file for Linux deployment"""
        print("🔧 Creating systemd service...")
        
        service_content = f'''[Unit]
Description=Face and Voice Recognition System
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory={self.project_root}
Environment=PATH={self.project_root}/venv/bin
ExecStart={self.project_root}/venv/bin/python app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
'''
        
        with open('face-voice-recognition.service', 'w') as f:
            f.write(service_content)
        
        print("✅ Systemd service file created")
        print("📝 To install: sudo cp face-voice-recognition.service /etc/systemd/system/")
        print("📝 To enable: sudo systemctl enable face-voice-recognition")
        print("📝 To start: sudo systemctl start face-voice-recognition")
    
    def backup_models(self):
        """Create backup of trained models"""
        print("💾 Creating model backup...")
        
        backup_dir = Path('model_backups')
        backup_dir.mkdir(exist_ok=True)
        
        model_files = ['face_model.h5', 'voice_model.h5', 'face_encoder.pkl', 'voice_encoder.pkl']
        backed_up = []
        
        for model_file in model_files:
            if Path(model_file).exists():
                shutil.copy2(model_file, backup_dir / model_file)
                backed_up.append(model_file)
        
        if backed_up:
            print(f"✅ Backed up: {', '.join(backed_up)}")
        else:
            print("ℹ️  No trained models found to backup")
    
    def generate_deployment_guide(self):
        """Generate deployment guide"""
        print("📚 Generating deployment guide...")
        
        guide = '''# Deployment Guide

## Local Development
1. Run: `python start.bat` (Windows) or `python app.py` (Linux/Mac)
2. Open: http://localhost:5000

## Production Deployment

### Option 1: Direct Python Deployment
1. Install Python 3.8+
2. Install requirements: `pip install -r requirements.txt`
3. Set environment variables:
   - `FLASK_ENV=production`
   - `SECRET_KEY=your-secret-key`
4. Run: `python app.py`

### Option 2: Docker Deployment
1. Build: `docker build -t face-voice-recognition .`
2. Run: `docker run -p 5000:5000 face-voice-recognition`

### Option 3: Docker Compose
1. Run: `docker-compose up -d`
2. Access: http://localhost

### Option 4: Systemd Service (Linux)
1. Copy service file: `sudo cp face-voice-recognition.service /etc/systemd/system/`
2. Enable: `sudo systemctl enable face-voice-recognition`
3. Start: `sudo systemctl start face-voice-recognition`

## Security Considerations
- Change default SECRET_KEY
- Use HTTPS in production
- Implement authentication
- Regularly backup trained models
- Monitor resource usage

## Performance Optimization
- Use GPU for model training
- Implement model caching
- Optimize image/audio preprocessing
- Use load balancing for multiple instances

## Monitoring
- Check logs: `journalctl -u face-voice-recognition`
- Monitor system resources
- Set up health checks
'''
        
        with open('DEPLOYMENT.md', 'w') as f:
            f.write(guide)
        
        print("✅ Deployment guide created")
    
    def run_deployment_setup(self):
        """Run complete deployment setup"""
        print("🚀 Face & Voice Recognition System - Deployment Setup")
        print("=" * 60)
        
        if not self.check_environment():
            print("❌ Environment check failed")
            return
        
        self.create_production_config()
        self.create_docker_files()
        self.create_systemd_service()
        self.backup_models()
        self.generate_deployment_guide()
        
        print("\n" + "=" * 60)
        print("✅ Deployment setup completed!")
        print("\n📁 Created files:")
        print("- prod_config.py (Production configuration)")
        print("- Dockerfile (Container setup)")
        print("- docker-compose.yml (Multi-container setup)")
        print("- nginx.conf (Reverse proxy config)")
        print("- face-voice-recognition.service (Systemd service)")
        print("- DEPLOYMENT.md (Deployment guide)")
        print("\n📚 Next steps:")
        print("1. Review DEPLOYMENT.md for deployment options")
        print("2. Test the application locally: python app.py")
        print("3. Choose your deployment method")
        print("4. Configure security settings")

def main():
    """Main deployment setup function"""
    deployment_manager = DeploymentManager()
    deployment_manager.run_deployment_setup()

if __name__ == "__main__":
    main()
