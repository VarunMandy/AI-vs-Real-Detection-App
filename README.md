# Real or AI? Human vs. Model

Small game I made for CS6180 HW1 at Northeastern. You get shown a face and have to guess if it's a real photo or AI-generated, then the model makes its guess too. 10 rounds, and at the end it shows who got more right.

Play it here: https://ai-vs-real-detection-app.streamlit.app/

## Model

I trained five models on 128x128 face images (MLP, a small CNN, the CNN with augmentation, MobileNetV3Small with a frozen backbone, and MobileNetV3Small fine-tuned). The fine-tuned MobileNet did best at about 76% test accuracy, so that's the one used in the game.

The test images in the game come from the held-out test set, so the model never saw them during training.

## How to play

Click Real or AI-generated for each face. After you guess it shows the right answer and what the model predicted. Hit Next image to keep going. After 10 rounds you get the final score and a recap of each round. New game starts over with different images.

## Files

- app.py - the Streamlit app
- best_model.keras - trained model
- config.json - model name, test accuracy and input scaling
- test_images/ - real and AI test images used in the game
- requirements.txt - dependencies

## Running it locally

    pip install -r requirements.txt
    streamlit run app.py
