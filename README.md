
# 🚀 Binance AI Trading Bot

A cutting-edge Binance AI Trading Bot that leverages advanced machine learning models to predict cryptocurrency prices and make smarter trading decisions. This project combines state-of-the-art time series forecasting models (LSTM, CNN-LSTM, Transformer, Prophet) with **sentiment analysis** from news articles, ensuring enhanced prediction accuracy.

---

## 📋 Table of Contents
1. [Features](#features)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Usage](#usage)
6. [Models Supported](#models-supported)
7. [Sentiment Analysis](#sentiment-analysis)
8. [Contributing](#contributing)
9. [License](#license)

---

## ✨ Features
- **Multiple Models**: Supports a variety of machine learning models for time series forecasting.
- **Real-Time Data**: Fetches live market data from Binance API.
- **Sentiment Analysis**: Integrates news sentiment analysis to adjust predictions dynamically.
- **Web Interface**: Features an intuitive Flask-based dashboard for user interaction.
- **Customization**: Fully configurable parameters for model selection, training, and prediction.

---

## 🏗️ Architecture
The bot is modular and well-organized, with the following structure:
```
binance-ai-trading-bot/
│
├── models/        # Machine learning models for predictions
├── data/          # Data loading and preprocessing scripts
├── utils/         # Utility functions for predictions and management
├── app_routes/    # Flask routes for the web dashboard
├── templates/     # HTML templates for web pages
└── config.py      # Configuration file for API keys and settings
```

---

## ⚙️ Installation

### Prerequisites
- **Python** (3.7 or higher)
- **Git**
- **Virtualenv** (recommended)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/binance-ai-trading-bot.git
   cd binance-ai-trading-bot
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🔧 Configuration

### Binance API Keys
1. Sign up on [Binance](https://www.binance.com) and obtain your **API key** and **Secret key**.
2. Add them to the `config.py` file or set them as environment variables.

### Modify Configurations
Adjust other settings in the `config.py` file as needed to suit your requirements.

---

## 🚀 Usage

### Running the Application
To start the application:
```bash
python app.py
```

### Accessing the Dashboard
Open your web browser and navigate to:
```
http://127.0.0.1:5000/
```

### Training a Model
1. Go to the **Train Model** page in the dashboard.
2. Select:
   - Model type
   - Coin pair (e.g., BTC/USDT)
   - Data period
   - Epochs and batch size
3. Click **Train** to start training.

### Making a Prediction
1. Navigate to the **Prediction** page.
2. Select a trained model from the dropdown.
3. (Optional) Enable **Sentiment Analysis**.
4. Click **Predict** to view the results.

---

## 🧠 Models Supported
- **LSTM**: Long Short-Term Memory
- **CNN-LSTM**: Hybrid Convolutional-LSTM Model
- **Transformer**: Attention-based Neural Network
- **Prophet**:High-performance solution for time-series forecasting

---

## 💬 Sentiment Analysis
- Fetches news articles related to the selected cryptocurrency.
- Uses **Natural Language Processing (NLP)** models to analyze sentiment.
- Dynamically adjusts price predictions based on market sentiment.

---

## 🤝 Contributing
Contributions are welcome! Please follow these steps:
1. **Fork** the repository.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature-name
   ```
3. Make your changes and commit them:
   ```bash
   git commit -m "Description of your changes"
   ```
4. Push your branch:
   ```bash
   git push origin feature-name
   ```
5. Submit a **Pull Request**.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE). Feel free to use and modify it as per the terms of the license.

---

Happy Trading! 🚀
