import socket
import json

server_ip = "127.0.0.1"  # This PC
server_port = 5000

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((server_ip, server_port))

print("Successfully Connected ！Input data to Estimate，input 'exit' to quit。")

while True:
    category = input("Choose category of prediction (A:weather and humidity in Capital Cities/B: air quality in California/C: S&P 500 stock price): ")
    if category.lower() == "exit":
        break
    user_message = input("You: ")
    if user_message.lower() == "exit":
        break

    # send JSON data
    request_data = json.dumps({"category": category, "message": user_message})
    client_socket.sendall(request_data.encode("utf-8"))

    # receive result
    response = client_socket.recv(1024).decode("utf-8")
    print(f"📡 Server response: {response}")

client_socket.close()
print("🔴 Connection outage")
