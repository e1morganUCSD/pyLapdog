import matplotlib.pyplot as plt
import cPickle as pickle

if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, "/Users/Eric/PycharmProjects/pyLapdog/")
    import contrastmodel.functions.masks as msk
    import contrastmodel.params.paramsDef as par

    params = par.Params()

    # # reduce the number of options so it processes more quickly
    # params.filt.orientations = range(0, 89, 30)
    # params.filt.stdev_pixels = [4.0, 8.0, 16.0]

    print("Generating masks...")
    msk.generate_correlation_mask_fft(params)

    print("loading filtermasks from file...")
    filtermasks = pickle.load(open("filtermasks_FFT.pkl", mode="rb"))

    print("loading ap_filtermasks from file...")
    ap_filtermasks = pickle.load(open("ap_filtermasks_FFT.pkl", mode="rb"))

    orientations = params.filt.orientations
    stdev_pixels = params.filt.stdev_pixels

    dir = os.path.dirname("output/")
    if not os.path.exists(dir):
        os.mkdir(dir)

    for o in range(len(orientations)):
        for f in range(len(stdev_pixels)):
            for o2 in range(len(orientations)):
                for f2 in range(len(stdev_pixels)):
                    tempmask = filtermasks[o][f][o2][f2]
                    tempmask_ap = ap_filtermasks[o][f][o2][f2]

                    filename = "sp-{}-{}-{}-{}.png".format(o, f, o2, f2)
                    ap_filename = "ap-{}-{}-{}-{}.png".format(o, f, o2, f2)
                    fig = plt.imshow(tempmask, interpolation="none")
                    plt.colorbar()
                    plt.suptitle(filename)
                    plt.savefig("output/" + filename)
                    plt.close()

                    fig = plt.imshow(tempmask_ap, interpolation="none")
                    plt.colorbar()
                    plt.suptitle(ap_filename)
                    plt.savefig("output/" + ap_filename)
                    plt.close()


