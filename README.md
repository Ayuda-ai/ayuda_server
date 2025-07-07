# Ayuda_Server

This repository contains backend code for Ayuda - an AI-powered course recommendation system using resume embeddings and semantic search.

## Table of Contents

- [Overview](#overview)
- [High-Level Design (HLD)](#high-level-design-hld)
- [Low-Level Design (LLD)](#low-level-design-lld)
- [Architecture](#architecture)
- [Services & Functionalities](#services--functionalities)
- [API Documentation](#api-documentation)
  - [Authentication](#authentication)
  - [User Management](#user-management)
  - [Course Management](#course-management)
  - [Recommendations](#recommendations)
  - [Admin Operations](#admin-operations)
  - [Neo4j Graph Operations](#neo4j-graph-operations)
- [Local Dev Environment Setup](#local-dev-environment-setup)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Performance Optimizations](#performance-optimizations)
- [Monitoring & Logging](#monitoring--logging)

## Overview

Ayuda is a sophisticated course recommendation system that leverages AI and machine learning to provide personalized course recommendations based on user profiles, resume analysis, and semantic similarity matching. The system integrates multiple technologies including FastAPI, PostgreSQL, Redis, Pinecone, Neo4j, and Ollama for LLM reasoning.

## High-Level Design (HLD)

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Mobile App    │    │   Third-party   │
│   (React/Vue)   │    │   (React Native)│    │   Integrations  │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │    FastAPI Backend       │
                    │   (Ayuda_Server)         │
                    └─────────────┬─────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
┌─────────▼─────────┐  ┌─────────▼─────────┐  ┌─────────▼─────────┐
│   PostgreSQL      │  │      Redis        │  │     Pinecone      │
│   (User Data)     │  │   (Caching)       │  │   (Embeddings)    │
└───────────────────┘  └───────────────────┘  └───────────────────┘
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │      Neo4j Graph         │
                    │   (Prerequisites)        │
                    └───────────────────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │      Ollama LLM          │
                    │   (Reasoning Engine)      │
                    └───────────────────────────┘
```

### Key Components:

1. **API Layer**: FastAPI-based RESTful APIs with JWT authentication
2. **User Management**: Registration, profile management, resume processing
3. **Course Management**: Course search, CRUD operations, major-based filtering
4. **Recommendation Engine**: Hybrid semantic + keyword matching with prerequisite checking
5. **Graph Database**: Neo4j for prerequisite relationships and course dependencies
6. **Vector Database**: Pinecone for semantic similarity search
7. **LLM Integration**: Ollama for personalized course reasoning
8. **Caching**: Redis for performance optimization
9. **Admin Panel**: Comprehensive system management and analytics

## Low-Level Design (LLD)

### Service Architecture:

```
app/
├── api/                    # API endpoints
│   ├── auth.py            # Authentication endpoints
│   ├── user.py            # User management endpoints
│   ├── admin.py           # Admin operations endpoints
│   ├── course.py          # Course search endpoints
│   ├── recommendations.py # Recommendation endpoints
│   └── neo4j.py          # Graph operations endpoints
├── services/              # Business logic services
│   ├── user_service.py    # User operations
│   ├── admin_service.py   # Admin operations
│   ├── course_service.py  # Course operations
│   ├── recommendation_service.py # Recommendation engine
│   ├── neo4j_service.py  # Graph database operations
│   ├── redis_service.py   # Caching operations
│   ├── llm_service.py    # LLM reasoning
│   └── recommendation_logger.py # Analytics logging
├── models/                # Database models
├── schemas/               # Pydantic schemas
├── core/                  # Core configurations
└── db/                    # Database configurations
```

### Data Flow:

1. **User Registration/Login** → JWT Token → Authenticated Requests
2. **Resume Upload** → Text Extraction → Embedding Generation → Storage
3. **Course Recommendation** → Profile Analysis → Semantic Search → Prerequisite Check → LLM Reasoning
4. **Admin Operations** → System Management → Analytics → Data Sync

## Architecture

### Technology Stack:

- **Framework**: FastAPI (Python)
- **Database**: PostgreSQL (Primary), Redis (Caching), Neo4j (Graph), Pinecone (Vector)
- **Authentication**: JWT with bcrypt
- **AI/ML**: Sentence Transformers, Scikit-learn, Ollama LLM
- **File Processing**: PyMuPDF, python-docx
- **Async Support**: httpx, asyncio
- **Monitoring**: Custom logging with JSON structured logs

### Performance Features:

- **Async Programming**: Non-blocking I/O operations
- **Connection Pooling**: Optimized database connections
- **Caching**: Redis-based response caching
- **Parallel Processing**: Concurrent API calls
- **Vector Search**: Semantic similarity matching
- **Graph Queries**: Efficient prerequisite checking

## Services & Functionalities

### Core Services:

1. **UserService**: User management, profile enhancement, resume processing
2. **AdminService**: System administration, analytics, data management
3. **RecommendationService**: Hybrid recommendation engine
4. **Neo4jService**: Graph database operations for prerequisites
5. **RedisService**: Caching and keyword matching
6. **LLMService**: AI-powered reasoning for recommendations
7. **CourseService**: Course management and search
8. **RecommendationLogger**: Analytics and performance monitoring

### Key Functionalities:

- **Resume Analysis**: PDF/DOCX parsing, text extraction, embedding generation
- **Semantic Search**: Vector-based course matching using Pinecone
- **Keyword Matching**: Redis-based skill matching
- **Prerequisite Checking**: Graph-based course dependency validation
- **LLM Reasoning**: Personalized course recommendation explanations
- **Profile Enhancement**: AI-powered skill and course suggestions
- **Admin Analytics**: Comprehensive system monitoring and statistics

## API Documentation

### Authentication

| Endpoint | Method | Description | Access |
|----------|--------|-------------|---------|
| `/auth/login` | POST | User login with JWT token generation | **USER** |

### User Management

| Endpoint | Method | Description | Access |
|----------|--------|-------------|---------|
| `/users/signup` | POST | User registration | **USER** |
| `/users/me` | GET | Get current user profile | **USER** |
| `/users/resume` | POST | Upload and process resume | **USER** |
| `/users/resume/embed` | POST | Generate resume embeddings | **USER** |
| `/users/resume/embedding` | GET | Get resume embedding | **USER** |
| `/users/resume` | DELETE | Delete resume data | **USER** |
| `/users/profile/completed-courses` | GET/POST/PUT/DELETE | Manage completed courses | **USER** |
| `/users/profile/additional-skills` | GET/PUT | Manage additional skills | **USER** |
| `/users/profile/enhancement-status` | GET | Get profile enhancement status | **USER** |
| `/users/profile/enhance` | POST | Enhance user profile | **USER** |

### Course Management

| Endpoint | Method | Description | Access |
|----------|--------|-------------|---------|
| `/courses/search` | GET | Search courses with filters | **USER** |
| `/courses/search/{course_id}` | GET | Get specific course details | **USER** |
| `/courses/major/{major}` | GET | Get courses by major | **USER** |

### Recommendations

| Endpoint | Method | Description | Access |
|----------|--------|-------------|---------|
| `/recommendations/match_courses` | GET | Hybrid course recommendations | **USER** |
| `/recommendations/semantic` | GET | Semantic-only recommendations | **USER** |
| `/recommendations/explain` | POST | Get reasoning for course recommendation | **USER** |
| `/recommendations/debug` | GET | Debug recommendation system | **USER** |
| `/recommendations/analytics` | GET | Recommendation analytics | **ADMIN** |
| `/recommendations/llm/health` | GET | LLM service health check | **USER** |

### Admin Operations

| Endpoint | Method | Description | Access |
|----------|--------|-------------|---------|
| `/admin/users` | GET | Get all users | **ADMIN** |
| `/admin/users/{user_id}` | GET/PUT/DELETE | Manage specific user | **ADMIN** |
| `/admin/courses` | GET/POST/PUT/DELETE | Manage courses | **ADMIN** |
| `/admin/system/health` | GET | System health check | **ADMIN** |
| `/admin/system/stats` | GET | System statistics | **ADMIN** |
| `/admin/system/sync/neo4j` | POST | Sync data to Neo4j | **ADMIN** |
| `/admin/system/sync/pinecone` | POST | Sync data to Pinecone | **ADMIN** |
| `/admin/system/backup` | POST | Create system backup | **ADMIN** |
| `/admin/system/restore` | POST | Restore system from backup | **ADMIN** |
| `/admin/parse_courses` | POST | Parse course data | **ADMIN** |

### Neo4j Graph Operations

| Endpoint | Method | Description | Access |
|----------|--------|-------------|---------|
| `/neo4j/health` | GET | Neo4j connection health | **USER** |
| `/neo4j/sync-courses` | POST | Sync courses to Neo4j | **ADMIN** |
| `/neo4j/prerequisites` | POST | Add prerequisite relationship | **ADMIN** |
| `/neo4j/prerequisites/{course_id}` | GET | Get course prerequisites | **USER** |
| `/neo4j/prerequisites/{course_id}/path` | GET | Get prerequisite path | **USER** |
| `/neo4j/prerequisites/check` | POST | Check prerequisite completion | **USER** |
| `/neo4j/courses/available` | POST | Get available courses | **USER** |
| `/neo4j/analytics` | GET | Course analytics | **ADMIN** |
| `/neo4j/courses/{course_id}/recommendations` | GET | Course-based recommendations | **USER** |

## Local Dev Environment Setup

- Install Python 3.11.2 and MongoDB
- Install all the Python packages used in this project using `pip install requirements.txt`
- Create a `db.ini` file in the root directory (where run.py exists). Use `db.ini.txt` file for the reference to create the `db.ini` file
- Refer this [link](https://www.prisma.io/dataguide/mongodb/connection-uris#:~:text=A%20quick%20description%20of%20each,username%20%3A%20An%20optional%20username.) to know the format of a standard MongoDB URI that needs to be put in the `db.ini`
- BASE_URL is url where you need the backend Flask server to run.It can be `http://127.0.0.1:5000`
- Once done with the above setup, start the server by `python run.py` command

## Installation

### Prerequisites

- Python 3.11.2+
- PostgreSQL 12+
- Redis 6+
- Neo4j 5+
- Ollama (for LLM reasoning)

### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd ayuda_server/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Run migrations
alembic upgrade head

# Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Database Connection
POSTGRES_DB=db_name
POSTGRES_USER=db_user
POSTGRES_PASSWORD=db_password
DATABASE_HOST=localhost
DATABASE_PORT=5432

# JWT Configs
JWT_SECRET_KEY=enter_jwt_secret_key_here
ACCESS_TOKEN_EXPIRE_MINUTES=10080

# Pinecone Credentials
PINECONE_API_KEY=pinecone_api_key_here
PINECONE_ENV=pinecone_env
PINECONE_INDEX=index_name

# Neo4j Credentials
NEO4J_URI=neo4j_uri
NEO4J_USERNAME=neo4j_username
NEO4J_PASSWORD=neo4j_password
AURA_INSTANCEID=neo4j_instance_id
AURA_INSTANCENAME=neo4j_instance_name

# Ollama Connection
OLLAMA_ADDRESS=http://localhost:11434/api/generate
OLLAMA_MODEL=model_name
```

## Running the Application

### Development Mode

```bash
# Start with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or use the provided script
./start_server.sh  # Linux/Mac
start_server.bat   # Windows
```

### Production Mode

```bash
# Using Docker
docker-compose up -d

# Or directly
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### API Documentation

Once running, access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Performance Optimizations

### Async Programming

The application uses async programming for:
- Database operations
- External API calls (Pinecone, Redis, Neo4j)
- File processing
- LLM reasoning

### Caching Strategy

- **Redis**: Session storage, keyword matching, response caching
- **Asynchronous Processing**: Asynchronous API calls

### Expected Performance

- **Response Time**: 2-4 seconds (optimized from 10+ seconds)
- **Concurrent Users**: 100+ simultaneous users
- **Recommendation Accuracy**: 85%+ relevance score

## Monitoring & Logging

### Logging

- **Structured Logging**: JSON format for easy parsing
- **Performance Metrics**: Response times, database queries
- **Error Tracking**: Comprehensive error logging
- **Analytics**: Recommendation effectiveness tracking

### Health Checks

- `/admin/system/health` - System health status
- `/recommendations/llm/health` - LLM service status
- `/neo4j/health` - Neo4j connection status

### Metrics

- User engagement analytics
- Recommendation accuracy metrics
- System performance monitoring
- Error rate tracking

---
