print("Initializing models..")
print("Initializing pipeline")
from transformers import pipeline
from flask import Flask, render_template
print("Initializing image feature extraction model..")
extraction = pipeline(task="image-feature-extraction", model="google/vit-base-patch16-224-in21k", use_fast=True)
print("Initialzing visual question answering model..")
qa = pipeline(task="visual-question-answering", model="Salesforce/blip-vqa-base", use_fast=True)
print("Models initialized; creating server")

