# 🕵️ Real or AI? — Human vs. Model

A game where you compete against a neural network to tell **real face photos** apart from
**AI-generated faces**. Each round shows one face from a held-out test set. You guess, then the
model reveals its prediction and confidence. After 10 rounds, whoever identified more faces
correctly wins.

**Play it here:** _add your Streamlit app URL_

## The model
Built for CS6180 HW1 at Northeastern University. Five models were trained and compared on
128×128 RGB face images: an MLP, a baseline CNN, a CNN with data augmentation, MobileNetV3Small
as a frozen feature extractor, and a fine-tuned MobileNetV3Small. The best model on the test set
is the one deployed here (its name and test accuracy are shown in the app).

## How to play
1. Look at the face and click **📷 Real** or **🤖 AI-generated**.
2. See the correct answer and what the model predicted.
3. Click **Next image** and repeat. After the last round you get the final score and a
   round-by-round recap. **🔄 New game** starts over with a fresh random set of images.

## Files
- `app.py`: Streamlit interface and game logic
- `best_model.keras`: trained Keras model
- `config.json`: model name, test accuracy, and input scaling
- `test_images/real`, `test_images/ai`: sample test images used in the game
- `requirements.txt`: Python dependencies

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
