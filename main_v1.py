def menu():
    print("[0] Exit")
    print("[1] Executing Word2Vec experiment...")
    print("[2] Executing Node2Vec experiment...")
    print("[3] Executing Graph2Vec experiment...")
    print("[4] Executing X experiment...")

    option = int(input("Enter your option: "))

    while option != 0:
        if option == 0:
            exit(0)
        if option == 1:
            print("[1] running")
        elif option == 2:
            print("[2] running")
        elif option == 3:
            print("[3] running")
        elif option == 4:
            print("[4] running")
        else:
            print("Invalid option.")
        print("\n \n")
        menu()
        option = int(input("Enter your option: "))


if __name__ == '__main__':
    menu()
