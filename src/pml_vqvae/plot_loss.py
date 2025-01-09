import yaml
import numpy as np
from pml_vqvae.visuals import violin_plot

if __name__ == "__main__":
    YAML_FILE = "artifacts/losses_per_class.yaml"

    with open(YAML_FILE, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    # calc mean
    for key, value in data.items():
        data[key] = {
            "mse": sum(value["mse"]) / len(value["mse"]),
            "ssim": sum(value["ssim"]) / len(value["ssim"]),
        }

    # mse list
    mse = [value["mse"] for value in data.values()]
    ssim = [value["ssim"] for value in data.values()]

    print("mean mse: ", np.mean(mse))
    print("mean ssim: ", np.mean(ssim))

    print("best mse: ", np.argmin(mse), " ", mse[np.argmin(mse)])
    print("best ssim: ", np.argmax(ssim), " ", ssim[np.argmax(ssim)])

    print("worst mse: ", np.argmax(mse), " ", mse[np.argmax(mse)])
    print("worst ssim: ", np.argmin(ssim), " ", ssim[np.argmin(ssim)])

    # print index of median mse
    mse = np.array(mse)
    print("median mse: ", np.argsort(mse)[len(mse) // 2])

    # print index of median ssim
    ssim = np.array(ssim)
    print("median ssim: ", np.argsort(ssim)[len(ssim) // 2])

    # violin_plot(
    #     [mse],
    #     ["MSE"],
    #     "MSE per class",
    #     ylabel="MSE",
    # )

    # violin_plot(
    #     [ssim],
    #     ["SSIM"],
    #     "SSIM per class",
    #     ylabel="SSIM",
    # )
