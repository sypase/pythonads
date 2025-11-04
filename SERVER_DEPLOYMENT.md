# Server Deployment Guide

## Issues Fixed for Large File Uploads

### 1. Uvicorn Configuration
Updated uvicorn settings to handle large file uploads:
- `--timeout-keep-alive 600` - 10 minute keep-alive timeout
- `--timeout-graceful-shutdown 30` - Graceful shutdown timeout
- `--limit-concurrency 1000` - Maximum concurrent connections
- `--backlog 2048` - Connection backlog
- `--limit-max-requests 10000` - Maximum requests per worker
- `--log-level info` - Detailed logging

### 2. Application Optimizations
- Files are processed in-memory (no disk I/O)
- Chunked async reading (1MB chunks)
- Enhanced error logging
- Progress tracking during upload

### 3. Testing Upload Issues

Use the `/test-upload` endpoint to diagnose upload problems:

```bash
curl -X POST "http://your-server:7777/test-upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your-file.pdf"
```

This will show:
- File size information
- Number of chunks received
- Total bytes received
- Any errors during upload

### 4. Common Server Issues (Without Nginx)

#### Cloud Provider Timeouts
If using AWS, GCP, Azure, or other cloud providers:
- Check load balancer timeout settings (usually 60 seconds default)
- Increase idle timeout to 600+ seconds
- Check if there's a proxy/load balancer in front

#### Firewall/Network Issues
- Ensure port 7777 is open
- Check firewall rules
- Verify network connectivity

#### Docker/Container Limits
- Check Docker memory limits: `docker stats`
- Ensure sufficient RAM for large files
- Check container logs: `docker compose logs -f`

### 5. Monitoring

Check logs in real-time:
```bash
docker compose logs -f fastapi
```

Look for:
- "Received file upload request" - Upload started
- "Read chunk: X bytes" - Progress
- "File reading complete" - Upload finished
- Any error messages

### 6. If Upload Still Fails

1. **Test with small file first** - Verify basic functionality
2. **Check server logs** - Look for timeout or connection errors
3. **Test with curl** - See raw response
4. **Check server resources** - CPU, memory, disk space
5. **Verify network** - Test from server itself vs external

### 7. Alternative: Use Gunicorn with Uvicorn Workers

If uvicorn alone isn't sufficient, you can use Gunicorn:

```bash
pip install gunicorn
```

Update docker-compose.yml command:
```yaml
command: gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:7777 --timeout 600 --graceful-timeout 30
```

