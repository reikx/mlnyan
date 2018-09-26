package mlnyan.ml;

import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class ModelBase {

    INDArray input;
    INDArray output;

    public abstract void init();

    public void setInput(INDArray input){
        this.input = input;
    }

    public INDArray getOutput(){
        return output;
    }

    public abstract String convertToJson();

    public abstract INDArray doBackPropagation(INDArray target);

    public abstract INDArray doBackPropagationByDiff(INDArray target);

    public abstract INDArray calc();

    public abstract int getInputSize();

    public abstract int getOutputSize();
}
