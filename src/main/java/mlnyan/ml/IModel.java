package mlnyan.ml;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface IModel {

    void init();

    void setInput(INDArray input);

    INDArray getOutput();

    String convertToJson();

    void doBackPropagation(INDArray target);

    void doBackPropagationByDiff(INDArray target);

    INDArray calc();
}
