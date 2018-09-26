package mlnyan.ml;

import org.nd4j.linalg.api.ndarray.INDArray;

public class PerceptronModel extends ModelBase{

    Layer[] layers;
    public PerceptronModel(Layer... layers){
        this.layers = layers;
        init();
    }

    @Override
    public void init(){
       /* for (int i = 0;i < layers.length - 1;++i){
            layers[i].weight = Nd4j.rand(layers[i + 1].perceptron.rows(),layers[i].perceptron.rows()).div(Math.sqrt(8 * layers[i].perceptron.rows()));
            layers[i].bias = Nd4j.zeros(layers[i + 1].perceptron.rows(),1);
        }*/
    }

    @Override
    public PerceptronModel clone(){
        Layer[] layers = new Layer[this.layers.length];
        for (int i = 0;i < layers.length;++i){
            layers[i] = this.layers[i].clone();
        }
        return new PerceptronModel(layers);
    }


    @Override
    public INDArray calc(){
        INDArray result = input;
        for (int i = 0;i < layers.length;++i){
            layers[i].setInput(result);
            result = layers[i].calc();
        }
        output = result;
        return result;
    }

    @Override
    public int getInputSize() {
        return layers[0].getInputSize();
    }

    @Override
    public int getOutputSize() {
        return layers[layers.length - 1].getOutputSize();
    }

    @Override
    public INDArray doBackPropagation(INDArray target){
        INDArray diff = output.sub(target);
        return doBackPropagationByDiff(diff);
    }

    public INDArray doBackPropagationByDiff(INDArray diff){
        for (int i = layers.length - 1;i >= 0;--i){
            diff = layers[i].bp(diff);
        }
        return diff;
    }


    @Override
    public String convertToJson(){
       /* ObjectMapper mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
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
        }*/
       return "";
    }


}
