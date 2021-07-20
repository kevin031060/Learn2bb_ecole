import matplotlib.pyplot as plt
problem = "auction"
def get_loss(log_name):
    trace_file = f"log/{problem}/{log_name}/{problem}_trace.txt"
    with open(trace_file, "r") as f:
        s = f.readlines()

    loss = [float(item.strip().split("val_loss: ")[-1].split(", kacc")[0]) for item in s]
    acc = [float(item.strip().split("val_accuracy: ")[-1].split(", val_loss")[0]) for item in s]
    return loss, acc

l1, a1 = get_loss(log_name = "20210719_1649")
l2, a2 = get_loss(log_name = "20210719_2225")
# l3, a3 = get_loss(log_name="20210717_0310")
plt.plot(l1[:140], label="pointer")
plt.plot(l2[:140], label="gnn")
# plt.plot(l3[:100], label="gnnm")
plt.legend()
plt.title("loss")
plt.show()

plt.plot(a1[:100], label="pointer")
plt.plot(a2[:100], label="gnn")
# plt.plot(a3[:100], label="gnnm")
plt.legend()
plt.title("accuracy")
plt.show()

