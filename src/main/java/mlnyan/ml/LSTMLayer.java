package mlnyan.ml;

import mlnyan.ml.LayerBase;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;

public class LSTMLayer extends LayerBase {
    LSTMBlock[] blocks;
    int size;
    public LSTMLayer(int size,LSTMBlock block){
        this.size = size;
        this.blocks = new LSTMBlock[size];
        for (int i = 0;i < size;++i){
            blocks[i] = block.initial();
        }
    }

    @Override
    public LayerBase initial() {
        return new LSTMLayer(size,blocks[0]);
    }

    @Override
    public INDArray calc() {
        INDArray ret = Nd4j.create(blocks.length,1);
        for (int i = 0;i < blocks.length;++i){
            ret.put(i,0,blocks[i].calc());
        }
        return ret;
    }

    @Override
    public void setInput(INDArray input) {
        for (int i = 0;i < input.rows();++i){
            blocks[i].setInput(input.getDouble(i,0));
        }
    }

    @Override
    public INDArray bp(INDArray diff) {
        INDArray retf = Nd4j.create(diff.rows(),1);
        for (int i = 0;i < diff.rows();++i){
            retf.put(i,0,blocks[i].doBackPropagationByDiff(diff.getDouble(i,0)));
        }
        return retf;
    }

    @Override
    protected LayerBase clone() {
        return null;
    }

    @Override
    public int getInputSize() {
        return size;
    }

    @Override
    public int getOutputSize() {
        return size;
    }

}
