package mlnyan.ml;

import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class LayerBase{
    public abstract LayerBase initial();

    public abstract INDArray calc();

    public abstract void setInput(INDArray input);

    public abstract INDArray bp(INDArray diff);

    @Override
    protected abstract LayerBase clone();

    public abstract int getInputSize();

    public abstract int getOutputSize();
}
