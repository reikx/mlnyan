package mlnyan.ml;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.SerializationFeature;
import org.nd4j.shade.jackson.databind.node.ArrayNode;
import org.nd4j.shade.jackson.databind.node.ObjectNode;

public class Model implements IModel{
    Layer[] layers;
    IBackPropagation backPropagation;
    public Model(IBackPropagation backPropagation,Layer... layers){
        this.layers = layers;
        this.backPropagation = backPropagation;
        init();
    }

    @Override
    public void init(){
        for (int i = 0;i < layers.length - 1;++i){
            layers[i].weight = Nd4j.rand(layers[i + 1].perceptron.rows(),layers[i].perceptron.rows()).div(Math.sqrt(8 * layers[i].perceptron.rows()));
            layers[i].bias = Nd4j.zeros(layers[i + 1].perceptron.rows(),1);
        }
    }

    @Override
    protected Model clone(){
        Layer[] layers = new Layer[this.layers.length];
        for (int i = 0;i < layers.length;++i){
            layers[i] = this.layers[i].clone();
        }
        return new Model(backPropagation,layers);
    }


    /* public void clear(){
        for (int i = 0;i < layers.length;++i){
            layers[i].perceptron = Nd4j.zeros(layers[i].perceptron.rows(),0);
        }
    } */

    @Override
    public INDArray calc(){
        for (int i = 0;i < layers.length - 2;++i){
            INDArray indArray = layers[i].weight.mmul(layers[i].perceptron).add(layers[i].bias);
            switch (layers[i].getActivator()){
                case SOFTMAX:layers[i + 1].perceptron = Transforms.softmax(indArray);break;
                case RELU:layers[i + 1].perceptron = Transforms.relu(indArray);break;
                default:layers[i + 1].perceptron = Transforms.relu(indArray);break;
            }
        }
        int t1 = layers.length - 2;
        int t2 = layers.length - 1;
        INDArray indArray = layers[t1].weight.mmul(layers[t1].perceptron).add(layers[t1].bias);
        switch (layers[t1].getActivator()){
            case SOFTMAX:layers[t2].perceptron = Transforms.softmax(indArray);break;
            case RELU:layers[t2].perceptron = Transforms.relu(indArray);break;
            case TANH:layers[t2].perceptron = Transforms.tanh(indArray);break;
            case SIGMOID:layers[t2].perceptron = Transforms.sigmoid(indArray);break;
            default: layers[t2].perceptron = Transforms.relu(indArray);
        }
        return layers[layers.length - 1].perceptron;
    }

    @Override
    public void doBackPropagation(INDArray target){
        backPropagation.calc(this,target);
    }

    public void doBackPropagationByDiff(INDArray dif){backPropagation.calcByDif(this,dif);}

    @Override
    public void setInput(INDArray input) {
        layers[0].perceptron = input.dup();
    }

    @Override
    public INDArray getOutput() {
        return layers[layers.length - 1].perceptron.dup();
    }


    @Override
    public String convertToJson(){
        ObjectMapper mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
        ArrayNode arr = mapper.createArrayNode();
        for (int i = 0;i < layers.length - 1;++i){
            ObjectNode node = mapper.createObjectNode();
            ArrayNode weight1 = mapper.createArrayNode();
            for (int y = 0; y < layers[i].weight.rows(); ++y) {
                ArrayNode weight2 = mapper.createArrayNode();
                for (int x = 0; x < layers[i].weight.columns(); ++x) {
                    weight2.add(layers[i].weight.getDouble(y, x));
                }
                weight1.add(weight2);
            }
            ArrayNode bias1 = mapper.createArrayNode();
            for (int y = 0; y < layers[i].bias.rows(); ++y) {
                bias1.add(layers[i].bias.getDouble(y));
            }
            node.set("weight", weight1);
            node.set("bias", bias1);
            arr.add(node);
        }
        ObjectNode node = mapper.createObjectNode();
        node.set("layers",arr);
        try {
            return mapper.writeValueAsString(node);
        } catch (JsonProcessingException e) {
            e.printStackTrace();
            return null;
        }
    }


}
