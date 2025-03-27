# Oracle_chat_bot_socket
📌 Project Overview
This project is an ELEC0088: Software for Network and Services Design 24/25 student assignment. It involves building a socket-based server and developing a chat program named Oracle Chat Bot. Users can interact with the chatbot by typing questions in the client, and the system will respond with machine learning model predictions in specific domains.

🔧 Features
The project consists of three main functionalities:

1️⃣ Feature A: Temperature & Humidity Prediction
Function Call: FuncA
Description: Predicts the temperature (°C & °F) and humidity for a given city on a specified date.
Supported Cities: Beijing, London, Washington, Paris, Berlin
2️⃣ Feature B: Air Quality Index (AQI) Prediction
Function Call: FuncB
Description: Predicts the AQI (Air Quality Index) for California based on different pollutant concentrations, including PM2.5, PM10, NOx, etc.
Supports: Querying AQI for different dates.
3️⃣ Feature C: S&P 500 Stock Price Prediction
Function Call: FuncC
Description: Predicts the S&P 500 stock price based on past market performance. The model provides multi-day price index forecasting.
🛠️ Installation & Execution
All functionalities are integrated into the main branch.
The project can be started with a one-click execution via the starter.bat file.
The bat file will launch the chatbot regardless of the installation path.
Important: The environment name inside the bat file must be updated based on the user's local setup.
📌 Notes
Ensure that all dependencies and the required environment are properly configured before execution.
The project uses socket programming for client-server communication.

Contribution:
            Yifei Li : server&client, Function A, readme file
            Cheng-Hsuan Hu : Function B train,modelling,verify,calling function
            Peiqi Xu : Function C train,modelling,verify
