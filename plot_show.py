import matplotlib.pyplot as plt

def get_loss(problem, log_name):
    trace_file = f"log/{problem}/{log_name}/{problem}_trace.txt"
    with open(trace_file, "r") as f:
        s = f.readlines()

    loss = [float(item.strip().split("val_loss: ")[-1].split(", kacc")[0]) for item in s]
    acc = [float(item.strip().split("val_accuracy: ")[-1].split(", val_loss")[0]) for item in s]
    return loss, acc

def plot_compare(problem, path1, path2, path3=None):

    l1, a1 = get_loss(problem, path1)
    l2, a2 = get_loss(problem, path2)
    if path3 is not None:
        l3, a3 = get_loss(problem, path3)

    plt.plot(l1, label="pointer")
    plt.plot(l2, label="gnn")
    if path3 is not None:
        plt.plot(l3[:100], label="gnnm")
    plt.legend()
    plt.title("loss")
    plt.show()

    plt.plot(a1, label="pointer")
    plt.plot(a2, label="gnn")
    if path3 is not None:
        plt.plot(a3[:100], label="gnnm")
    plt.legend()
    plt.title("accuracy")
    plt.show()

if __name__ == '__main__':
    plot_compare("indset", "20210723_2237", "20210724_0116")

    plot_compare("auction", "20210723_2236", "20210724_0221", "20210724_0426")

