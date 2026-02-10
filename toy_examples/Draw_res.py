import matplotlib.pyplot as plt
import numpy as np


def draw_layernums():
    nice_f_mse_1 = [2.8814304, 0.0910656, 0.00956122, 0.01414816, 0.00381014, 0.00236554]
    nice_f_mse_2 = [0.0676754, 1.5548428, 0.0065844, 0.01168255, 0.00429849, 0.00778353]
    nice_i_mse_1 = [3.5527137e-16, 5.4798005e-15, 2.2666843e-13, 2.3876058e-14, 1.0194349e-13,1.8245616e-13]
    nice_i_mse_2 = [2.3772651e-14, 6.8902092e-14, 4.0278317e-14, 4.8207376e-14, 4.2171632e-15, 2.0558379e-14]

    iresnet_f_mse_1 = [0.6699568, 0.5851129, 0.5198286, 0.43060455, 0.34190628, 0.42690077]
    iresnet_f_mse_2 = [0.3975359, 0.3175275, 0.32810935, 0.35178837, 0.34913376, 0.32778385]
    iresnet_i_mse_1 = [None, 3.0821658e+28, 2.4774003e-13, 1.4997989, 0.00396754, 1.5695779]
    iresnet_i_mse_2 = [None, 1.5718039e+28, 6.8935051e-14, 4.750672, 0.0036427, 3.8712893]

    xl = [1, 2, 3, 4, 5, 6]

    plt.figure()
    plt.plot(xl, nice_f_mse_1, marker = "o", label = "nice_f_mse_1", color = "blue")
    plt.plot(xl, nice_f_mse_2, marker = "o", label = "nice_f_mse_2", color = "blue")
    plt.plot(xl, iresnet_f_mse_1, marker = "o", label = "iresnet_f_mse_1", color = "orange")
    plt.plot(xl, iresnet_f_mse_2, marker = "o", label = "iresnet_f_mse_2", color = "orange")
    plt.legend()
    plt.xlabel("layer number")
    plt.ylabel("MSE")
    plt.title("Forward performance")

    plt.figure()
    plt.plot(xl, nice_i_mse_1, marker = "o", label = "nice_i_mse_1", color = "blue")
    plt.plot(xl, nice_i_mse_2, marker = "o", label = "nice_i_mse_2",  color = "blue")
    plt.plot(xl, iresnet_i_mse_1, marker = "o", label = "iresnet_i_mse_1",  color = "orange")
    plt.plot(xl, iresnet_i_mse_2, marker = "o", label = "iresnet_i_mse_2", color = "orange")
    plt.legend()
    plt.xlabel("layer number")
    plt.ylabel("MSE")
    plt.title("Inverse performance for different layer number")
    plt.show()



def draw_datadims():
    xl = [2, 4, 8, 16, 32, 64]
    nice_f_mse = [0.0005072116618975997, 0.0008067486342042685, 0.0005510819028131664, 0.00048721133498474956, 0.0006839548004791141, 0.0005673644482158124]
    nice_i_mse = [4.205178075872805e-15, 2.538713596983433e-15, 2.7203176726329917e-15, 2.316153996026471e-15, 1.8013896276069304e-15, 2.3568685404603803e-15]

    iresnet_f_mse = [0.0005111208301968873, 0.0005851120222359896, 0.00025890496908687055, 0.00016294252418447286, 0.00015430324128828943, 0.00014179338177200407]
    iresnet_i_mse = [2.5941679970072247e-14, 1.045650892261829e-11, 0.0007724933093413711, 0.0007314521353691816, 5.8818077377509326e-05, 0.0004432270070537925]


    plt.figure()
    plt.plot(xl, nice_f_mse, marker="o", label="nice_f_mse", color="blue")
    plt.plot(xl, iresnet_f_mse, marker="o", label="iresnet_f_mse", color="orange")
    plt.legend()
    plt.title("Forward performance")
    plt.xlabel("data dims")
    plt.ylabel("MSE")

    plt.figure()
    plt.plot(xl, nice_i_mse, marker="o", label="nice_i_mse_1", color="blue")
    plt.plot(xl, iresnet_i_mse, marker="o", label="iresnet_i_mse_1", color="orange")
    plt.legend()
    plt.title("Inverse performance")
    plt.xlabel("data dims")
    plt.ylabel("MSE")
    plt.show()




def layer_num_vs_fwd_mse():
    layers = np.array([2, 4, 6, 8, 10, 12])
    # fig_name = "layernum_fwdmse_toy.png"
    # mse_resnet = np.array(
    #     [0.004593844525516033, 0.0008088175090961158, 0.000624895328655839, 0.0008191089145839214, 0.0009413563529960811,
    #      0.0005548297776840627])
    # mse_iresnet = np.array(
    #     [1.398442268371582, 0.856304407119751, 0.44413307309150696, 0.16272762417793274, 0.0458192452788353,
    #      0.004368127789348364])
    # mse_dipdnn = np.array(
    #     [0.052817754447460175, 0.017798133194446564, 0.002549456652923822, 0.0007939437055028975, 0.0009919957956299186,
    #      0.0008242417825385928])
    # mse_nice = np.array(
    #     [0.13927645981311798, 0.005859000608325005, 0.0009767223382368684, 0.0007206216687336564, 0.0003909909864887595,
    #      0.0003871423832606524])

    fig_name = "layernum_fwdmse_powerflow.png"
    # ResNet accuracy values
    mse_resnet = np.array([0.0005835551419295371, 0.0002812569728121615, 0.00019949450506828725,
                           0.00014142406871542335, 5.106455864734016e-05, 3.9873822970548645e-05])
    # iResNet accuracy values
    mse_iresnet = np.array([0.050531331449747086, 0.0378844179213047, 0.02544865384697914,
                            0.012209424749016762, 0.0024811699986457825, 0.00016344636969733983])
    # DipDNN accuracy values
    mse_dipdnn = np.array([0.0013728757621720433, 0.0006398205296136439, 0.00031139873317442834,
                           0.00012377198436297476, 0.0001826223888201639, 0.0005454652127809823])
    # DipDNN accuracy values
    mse_nice = np.array([0.35405975580215454, 0.0021051198709756136, 0.0011913699563592672, 0.001509023248218
                            , 0.0011289413087069988, 0.0012389475014060736])

    # Plotting the data
    plt.figure(figsize=(8, 6))
    plt.plot(layers, mse_resnet, label='ResNet', marker='o')
    plt.plot(layers, mse_iresnet, label='iResNet', marker='o')
    plt.plot(layers, mse_dipdnn, label='DipDNN', marker='o')
    plt.plot(layers, mse_nice, label='NICE', marker='o')

    # Setting logarithmic scale for y-axis
    plt.yscale('log')

    # Labeling the axes
    plt.xlabel('Number of Layers')
    plt.ylabel('MSE (log scale)')

    # Adding title and legend
    plt.title('MSE vs Number of Layers for Different Models')
    plt.legend()
    plt.savefig(f"/Users/muhaoguo/Documents/study/Paper_Projects/Inverse_learning/Inverse_learning/toy_examples/{fig_name}")

    # Displaying the plot
    plt.show()

if __name__ == "__main__":
    layer_num_vs_fwd_mse()
