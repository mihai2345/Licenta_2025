from src.models.simple_nn import SimpleNN

def main():
    model = SimpleNN(input_dim=2, hidden_dim=4, output_dim=1)
    print(model)

if __name__ == "__main__":
    main()