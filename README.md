# <div align="center">Multimodal Sentiment and Stance Detection</div>

![Project Banner](https://via.placeholder.com/800x200.png?text=Multimodal+Sentiment+and+Stance+Detection)

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Welcome to the Multimodal Sentiment and Stance Detection project! This tool is designed to analyze videos and determine the sentiment and stance of individual speakers, as well as the overall tone of the conversation.

## Features
- **Multimodal Analysis**: Combines visual and audio data to provide accurate sentiment and stance detection.
- **Speaker-specific Results**: Analyzes each speaker individually and provides detailed results.
- **Global Analysis**: Provides an overall analysis of the conversation, including host bias if present.

## Installation
To get started with this project, follow the steps below:

1. **Download the Singularity Image**
   - Download the Singularity image from [this link](#) and copy it into the project folder.

2. **Add Videos**
   - Place the videos you want to analyze into the `Videos` folder.

## Usage
To run the analysis, execute the following command in the terminal:

```bash
sbatch template_segmentation.slurm "Gun Control"
