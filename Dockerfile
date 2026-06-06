# TalentLens - reproducible image. By default it ranks the FULL candidate pool:
# the entrypoint downloads candidates.jsonl from Drive and ranks every candidate.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gdown

COPY rank.py entrypoint.sh ./
COPY jobs ./jobs
COPY sample_candidates.json .
RUN chmod +x entrypoint.sh

# Pre-download the embedding model so ranking itself needs no network.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Public Drive link to the full candidates.jsonl (override with -e DRIVE_FILE_URL=...).
ENV DRIVE_FILE_URL="https://drive.google.com/file/d/1Fj2VEfC6mOHYIxuPsQRLraaBxglERc9P/view" \
    TOPK=100

# Default: download + rank ALL candidates, writing /data/submission.csv.
#   docker run --rm -v $PWD/out:/data talentlens
# Quick 50-candidate smoke test (no download):
#   docker run --rm -e RANK_SAMPLE=1 talentlens
# Use your own local file instead of downloading:
#   docker run --rm -v $PWD/data:/data -e CANDIDATES=/data/candidates.jsonl talentlens
ENTRYPOINT ["./entrypoint.sh"]
