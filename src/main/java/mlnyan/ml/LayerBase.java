package mlnyan.ml;

import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class LayerBase{
    public abstract LayerBase initial();

    public abstract INDArray bp(INDArray diff);
}
