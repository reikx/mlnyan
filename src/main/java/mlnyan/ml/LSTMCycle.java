package mlnyan.ml;

import mlnyan.sys.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;

public class LSTMCycle{
    LSTMModel base;

    ArrayList<LSTMModel> models = new ArrayList<>();

    public LSTMCycle(LSTMBlock lstmBlock,double learning,Layer... layers){
        this.base = new LSTMModel(lstmBlock,learning,layers);
        models.add(base);
    }

    public LSTMModel getCurrentModel(){
        return models.get(models.size() - 1);
    }

    public void next(){
        models.add(getCurrentModel().next());
    }

    public void bp(INDArray[] target){
        if(target.length != models.size()){
            Logger.warn(target.length  + "/" + models.size());
            //return;
        }
        LSTMModel total = base.clone();
        for (int i = models.size() - 2;i >= 0;--i){
            LSTMModel mb1 = models.get(i).clone();
            LSTMModel mb2 = models.get(i).clone();
            mb2.doBackPropagation(target[i]);
            for (int j = 0;j < mb2.layers.length;++j){
                Layer layer1 = mb1.layers[j];
                Layer layer2 = mb2.layers[j];
                total.layers[j].weight.add(layer2.weight.sub(layer1.weight));
                total.layers[j].bias.add(layer2.bias.sub(layer1.bias));
            }
            for (int j = 0;j < mb2.lstmLayers.length;++j){
                LSTMLayer llayer1 = mb1.lstmLayers[j];
                LSTMLayer llayer2 = mb2.lstmLayers[j];
                for (int k = 0;k < llayer1.blocks.length;++k){
                    LSTMBlock block1 = llayer1.blocks[k];
                    LSTMBlock block2 = llayer2.blocks[k];
                    {
                        PerceptronModel model1 = block1.sigmoid1;
                        PerceptronModel model2 = block2.sigmoid1;
                        for (int p = 0;p < model1.layers.length;++p){
                            Layer layer1 = model1.layers[p];
                            Layer layer2 = model2.layers[p];
                            total.lstmLayers[i].blocks[j].sigmoid1.layers[p].weight.add(layer2.weight.sub(layer1.weight));
                            total.lstmLayers[i].blocks[j].sigmoid1.layers[p].bias.add(layer2.bias.sub(layer1.bias));
                        }
                    }
                    {
                        PerceptronModel model1 = block1.sigmoid2;
                        PerceptronModel model2 = block2.sigmoid2;
                        for (int p = 0;p < model1.layers.length;++p){
                            Layer layer1 = model1.layers[p];
                            Layer layer2 = model2.layers[p];
                            total.lstmLayers[i].blocks[j].sigmoid2.layers[p].weight.add(layer2.weight.sub(layer1.weight));
                            total.lstmLayers[i].blocks[j].sigmoid2.layers[p].bias.add(layer2.bias.sub(layer1.bias));
                        }
                    }
                    {
                        PerceptronModel model1 = block1.sigmoid3;
                        PerceptronModel model2 = block2.sigmoid3;
                        for (int p = 0;p < model1.layers.length;++p){
                            Layer layer1 = model1.layers[p];
                            Layer layer2 = model2.layers[p];
                            total.lstmLayers[i].blocks[j].sigmoid3.layers[p].weight.add(layer2.weight.sub(layer1.weight));
                            total.lstmLayers[i].blocks[j].sigmoid3.layers[p].bias.add(layer2.bias.sub(layer1.bias));
                        }
                    }
                    {
                        PerceptronModel model1 = block1.tanh;
                        PerceptronModel model2 = block2.tanh;
                        for (int p = 0;p < model1.layers.length;++p){
                            Layer layer1 = model1.layers[p];
                            Layer layer2 = model2.layers[p];
                            total.lstmLayers[i].blocks[j].tanh.layers[p].weight.add(layer2.weight.sub(layer1.weight));
                            total.lstmLayers[i].blocks[j].tanh.layers[p].bias.add(layer2.bias.sub(layer1.bias));
                        }
                    }
                }
            }
        }
        base = total;
        models = new ArrayList<>();
        models.add(base);
    }
}
