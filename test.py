import time

def main():
    while True:
        print("t")
        time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Shutting down")