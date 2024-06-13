<div align="center">

<h1>Stable Diffusion 3 | Worker</h1>

🚀 | RunPod implementation of SD3-medium for serverless deployment.
</div>

## 📖 | Getting Started

### To make updates:

1. Clone this repository.
2. Obtain access to the Stable Diffusion 3 model on Huggingface
3. Build a container using token in the command: `HF_TOKEN=<YOUR_TOKEN> docker build --secret id=HF_TOKEN .`
4. Push it to your DockerHub account
5. Create a new serverless endpoint on RunPod with the image you just pushed.

## 🔗 | Links
- 🐳 [Docker Container](https://hub.docker.com/r/runpod/ai-api-sdxl)
- [RunPod workers repo](https://github.com/runpod-workers)
