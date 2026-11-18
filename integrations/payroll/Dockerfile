# The payroll agent's image: the OpenHands environment (integrations/openhands/Dockerfile) plus what upstream's task
# image gives the agent for spreadsheets and PDFs (slm-harness-adaptation's tasks/Dockerfile):
#   docker build -t honing-openhands integrations/openhands
#   docker build -t honing-payroll integrations/payroll
# Only the agent runs here. The grader runs on the host (integrations/payroll/grade.py), and the ground truth is never
# mounted.
FROM honing-openhands

RUN apt-get update && apt-get install -y --no-install-recommends jq poppler-utils \
 && rm -rf /var/lib/apt/lists/*
RUN uv pip install --system --no-cache "pandas>=2.3.3" "openpyxl>=3.1.5" numpy
