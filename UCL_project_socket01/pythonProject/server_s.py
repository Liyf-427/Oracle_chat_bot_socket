import socket
import diy as d  # get `d.main()` to proceed estimation from diy function lib
import json

# create Socket server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(("0.0.0.0", 5000))  # listen all IP
server_socket.listen(5)  # Maximize 5 client connection

print("✅ Server start，waiting connection...")

def handle_client(client_socket):
    """ process request """
    while True:
        try:
            # receive data from client
            data = client_socket.recv(1024).decode("utf-8")
            if not data:
                break

            print(f"Receive Input Data: {data}")

            # load JSON data
            try:
                request_data = json.loads(data)
                category = request_data.get("category", "")
                input_data = request_data.get("message", "")

                # use `d.main()` to predict
                prediction = d.main(category, input_data)
                response = d.print_result(category, input_data, prediction)
            except Exception as e:
                response = f"Error: {str(e)}"

            # send prediction result
            client_socket.sendall(response.encode("utf-8"))
            print(f"Sending Prediction: {response}")

        except Exception as e:
            print(f"❌ ERROR: {e}")
            break

    client_socket.close()
    print("🔴 Client offline")

# listen to linkage of client
while True:
    client_socket, addr = server_socket.accept()
    print(f"🔗Clint {addr} Connection Successfully")
    handle_client(client_socket)
