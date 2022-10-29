from geo import GeoModel, read_from_file

import pyro


if __name__ == "__main__":
    # read (from stdin) the location of the input file
    print("[input]: Please enter the relative path to the input file...")
    file_location = input()

    print("[input]: Please enter the number of steps...")
    num_of_steps = int(input())

    # load the observations from the input file
    heads, reference_times, observed_deformations = read_from_file(
        file_location)

    # initialize the model
    model = GeoModel(heads, reference_times, observed_deformations)

    # perform inference
    auto_guide = pyro.infer.autoguide.AutoNormal(
        pyro.poutine.block(model.model, hide=['nclay']))
    adam = pyro.optim.Adam({"lr": 0.02})  # Consider decreasing learning rate.
    elbo = pyro.infer.Trace_ELBO()
    svi = pyro.infer.SVI(model.model, auto_guide, adam, elbo)

    losses = []
    for step in range(num_of_steps):
        loss = svi.step()
        losses.append(loss)

    print("\nLosses:", losses, "\n")

    print("Inferred latent variables:")
    for key, val in auto_guide.median().items():
        print("   ", key + ":", val.item())
    predictive = pyro.infer.Predictive(model.model,
                                       guide=auto_guide,
                                       num_samples=10)

    pred_nclay = predictive()["nclay"].mode().values.item()
    print("   ", "nclay:", pred_nclay)
