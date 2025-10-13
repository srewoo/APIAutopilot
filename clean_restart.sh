#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}    🧹 API Autopilot V2 - Clean Restart Script 🚀${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

# Function to kill process on port
kill_port() {
    local port=$1
    local name=$2
    echo -e "${YELLOW}⚡ Stopping $name on port $port...${NC}"
    lsof -ti:$port | xargs kill -9 2>/dev/null || echo -e "${GREEN}✓ Port $port is free${NC}"
}

# 1. Stop all running servers
echo -e "${YELLOW}🛑 Stopping all servers...${NC}"
kill_port 8000 "Backend Server"
kill_port 3000 "Frontend Server"
sleep 2

# 2. Clear backend caches
echo -e "${YELLOW}🧹 Clearing backend caches...${NC}"

# Clear Python cache
find backend -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find backend -type f -name "*.pyc" -delete 2>/dev/null
echo -e "${GREEN}✓ Python cache cleared${NC}"

# Clear log files
rm -f backend/*.log 2>/dev/null
rm -f backend.log 2>/dev/null
echo -e "${GREEN}✓ Log files cleared${NC}"

# Clear and restart Redis cache if Redis is installed
if command -v redis-cli &> /dev/null; then
    redis-cli FLUSHALL 2>/dev/null && echo -e "${GREEN}✓ Redis cache cleared${NC}" || echo -e "${YELLOW}⚠ Redis not running (skipped)${NC}"

    # Restart Redis
    echo -e "${YELLOW}🔄 Restarting Redis...${NC}"
    if command -v brew &> /dev/null; then
        # macOS with Homebrew
        brew services restart redis 2>/dev/null && echo -e "${GREEN}✓ Redis restarted${NC}" || echo -e "${YELLOW}⚠ Redis restart failed (trying manual start)${NC}"
    elif command -v systemctl &> /dev/null; then
        # Linux with systemd
        sudo systemctl restart redis 2>/dev/null && echo -e "${GREEN}✓ Redis restarted${NC}" || echo -e "${YELLOW}⚠ Redis restart failed${NC}"
    else
        # Manual start
        redis-server --daemonize yes 2>/dev/null && echo -e "${GREEN}✓ Redis started${NC}" || echo -e "${YELLOW}⚠ Redis manual start failed${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Redis not installed (skipped)${NC}"
fi

# Clear MongoDB test data (optional - uncomment if needed)
# mongo api_autopilot --eval "db.test_results.remove({}); db.api_specs.remove({})" 2>/dev/null && \
#     echo -e "${GREEN}✓ MongoDB test data cleared${NC}" || \
#     echo -e "${YELLOW}⚠ MongoDB not running (skipped)${NC}"

# 3. Clear frontend caches
echo -e "${YELLOW}🧹 Clearing frontend caches...${NC}"

# Clear npm cache
cd frontend 2>/dev/null && {
    rm -rf node_modules/.cache 2>/dev/null
    rm -rf build 2>/dev/null
    rm -rf .next 2>/dev/null
    rm -f package-lock.json 2>/dev/null
    echo -e "${GREEN}✓ Frontend cache cleared${NC}"

    # Reinstall dependencies if needed
    if [ ! -d "node_modules" ]; then
        echo -e "${YELLOW}📦 Installing frontend dependencies...${NC}"
        npm install --silent
        echo -e "${GREEN}✓ Dependencies installed${NC}"
    fi
} || echo -e "${YELLOW}⚠ Frontend directory not found${NC}"

cd ..

# 4. Start Backend Server
echo ""
echo -e "${BLUE}🚀 Starting Backend Server (V2)...${NC}"
cd backend

# Check virtual environment
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}📦 Creating Python virtual environment...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Virtual environment ready${NC}"
else
    source venv/bin/activate
fi

# Start backend with V2 server
if [ -f "server_v2.py" ]; then
    echo -e "${GREEN}✓ Using API Autopilot V2 (Enhanced)${NC}"
    nohup uvicorn server_v2:app --reload --host 0.0.0.0 --port 8000 > ../backend.log 2>&1 &
    BACKEND_PID=$!
    echo -e "${GREEN}✓ Backend V2 started (PID: $BACKEND_PID)${NC}"
else
    echo -e "${RED}✗ server_v2.py not found!${NC}"
    exit 1
fi

cd ..

# 5. Start Frontend Server
echo -e "${BLUE}🚀 Starting Frontend Server...${NC}"
cd frontend 2>/dev/null && {
    # Check if package.json exists
    if [ -f "package.json" ]; then
        # Set environment variable to use V2 endpoints
        export REACT_APP_USE_V2=true
        export REACT_APP_BACKEND_URL=http://localhost:8000

        nohup npm start > ../frontend.log 2>&1 &
        FRONTEND_PID=$!
        echo -e "${GREEN}✓ Frontend started (PID: $FRONTEND_PID)${NC}"
    else
        echo -e "${YELLOW}⚠ Frontend package.json not found${NC}"
    fi
} || echo -e "${YELLOW}⚠ Frontend directory not found${NC}"

cd ..

# 6. Wait for services to be ready
echo ""
echo -e "${YELLOW}⏳ Waiting for services to start...${NC}"
sleep 5

# 7. Health check
echo -e "${BLUE}🔍 Checking service health...${NC}"

# Check backend
if curl -s http://localhost:8000/api/v2/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Backend V2 is healthy${NC}"
    BACKEND_HEALTH=$(curl -s http://localhost:8000/api/v2/health)
    echo -e "  Status: $(echo $BACKEND_HEALTH | grep -o '"status":"[^"]*"' | cut -d'"' -f4)"
    echo -e "  MongoDB: $(echo $BACKEND_HEALTH | grep -o '"mongodb":[^,}]*' | cut -d':' -f2)"
    echo -e "  Redis: $(echo $BACKEND_HEALTH | grep -o '"redis":[^,}]*' | cut -d':' -f2)"
else
    echo -e "${RED}✗ Backend is not responding${NC}"
fi

# Check frontend
if curl -s http://localhost:3000 > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Frontend is running${NC}"
else
    echo -e "${YELLOW}⚠ Frontend is starting (may take a moment)${NC}"
fi

# 8. Show available endpoints
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}🎉 Clean Restart Complete!${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${GREEN}Available Services:${NC}"
echo -e "  📡 Backend API: ${BLUE}http://localhost:8000${NC}"
echo -e "  📚 API Docs: ${BLUE}http://localhost:8000/docs${NC}"
echo -e "  🎨 Frontend: ${BLUE}http://localhost:3000${NC}"
echo ""
echo -e "${GREEN}V2 Endpoints:${NC}"
echo -e "  • POST /api/v2/generate-tests"
echo -e "  • POST /api/v2/discover-api"
echo -e "  • POST /api/v2/execute-tests"
echo -e "  • POST /api/v2/security-scan"
echo -e "  • GET  /api/v2/test-profiles"
echo -e "  • GET  /api/v2/frameworks"
echo -e "  • GET  /api/v2/health"
echo ""
echo -e "${YELLOW}Logs:${NC}"
echo -e "  • Backend: tail -f backend.log"
echo -e "  • Frontend: tail -f frontend.log"
echo ""
echo -e "${YELLOW}To stop all services:${NC}"
echo -e "  • Run: ${BLUE}./stop_all.sh${NC} (or manually kill ports 3000 and 8000)"
echo ""