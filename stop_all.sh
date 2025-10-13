#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${RED}    🛑 Stopping API Autopilot Services${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

# Function to kill process on port
kill_port() {
    local port=$1
    local name=$2
    echo -e "${YELLOW}⚡ Stopping $name on port $port...${NC}"

    # Get PID of process using the port
    PID=$(lsof -ti:$port 2>/dev/null)

    if [ ! -z "$PID" ]; then
        kill -9 $PID 2>/dev/null
        echo -e "${GREEN}✓ Stopped $name (PID: $PID)${NC}"
    else
        echo -e "${GREEN}✓ $name was not running${NC}"
    fi
}

# Stop Backend Server
kill_port 8000 "Backend Server"

# Stop Frontend Server
kill_port 3000 "Frontend Server"

# Optional: Stop Redis
if command -v redis-cli &> /dev/null; then
    redis-cli ping > /dev/null 2>&1 && {
        echo -e "${YELLOW}⚡ Stopping Redis cache...${NC}"
        redis-cli shutdown 2>/dev/null || echo -e "${YELLOW}⚠ Redis will continue running (used by other apps)${NC}"
    } || echo -e "${GREEN}✓ Redis was not running${NC}"
fi

# Optional: Stop MongoDB
# Uncomment if you want to stop MongoDB as well
# if command -v mongod &> /dev/null; then
#     echo -e "${YELLOW}⚡ Stopping MongoDB...${NC}"
#     mongod --shutdown 2>/dev/null || echo -e "${YELLOW}⚠ MongoDB will continue running (used by other apps)${NC}"
# fi

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ All API Autopilot services stopped successfully!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}To restart services:${NC}"
echo -e "  • Clean restart: ${BLUE}./clean_restart.sh${NC}"
echo -e "  • Normal restart: ${BLUE}./restart.sh${NC}"
echo ""