class GeoModel:
    def __init__(self, heads, reference_times, observed_deformations):
        # initializations
        
        # sampling
        self.draw()

    def draw(self):
        pass
        # sampling

    def run(self):
        pass
        # run inference


def read_from_file(file_location: str):
    """Read the input data from the given file

    Args:
        file_location (str): The location of the given input file.

    Returns:
        (List[int], List[int], List[int]): Three lists of the same length,
            corresponding to: heads, reference time and observed deformation
            measurments.
    """
    with open(file_location) as f:
        heads = [float(h) for h in f.readline().split()]
        reference_times = [float(h) for h in f.readline().split()]
        observed_deformations = [float(h) for h in f.readline().split()]
    if len(heads) != len(reference_times) != len(observed_deformations):
        raise RuntimeError("Heads, reference times and observed deformations"
                           " were expected to have the same length but they"
                           " didn't.")
    return heads, reference_times, observed_deformations


if __name__ == "__main__":
    # read (from stdin) the location of the input file
    file_location = input()

    # load the inputs from the input file
    heads, reference_times, observed_deformations = read_from_file(
        file_location)

    # initialize the model
    model = GeoModel(heads, reference_times, observed_deformations)

    # perform inference
    model.run()