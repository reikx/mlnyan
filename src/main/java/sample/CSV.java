package sample;

import mlnyan.sys.Logger;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class CSV {
    HashMap<String,ArrayList<String>> map = new HashMap<>();

    public String get(String column,int index){
        if(!map.containsKey(column)){
            Logger.warn("csv does not have such column data -> " + column);
            return null;
        }
        ArrayList<String> s = map.get(column);
        if(index < 0||s.size() <= index){
            Logger.warn("illegal index value");
            return null;
        }
        return s.get(index);
    }

    public ArrayList<String> getArrayList(String column){
        if(!map.containsKey(column)){
            Logger.warn("csv does not have such column data -> " + column);
            return null;
        }
        return map.get(column);
    }

    public int size(String column){
        return map.containsKey(column) ? map.get(column).size() : 0;
    }

    public static CSV load(String filePath){
        return load(new File(filePath));
    }

    public static CSV load(File file){
        CSV csv = new CSV();
        if(!file.exists())return csv;
        try {
            BufferedReader reader = new BufferedReader(new FileReader(file));
            String s = reader.readLine();
            if(s == null)return csv;
            String[] column = parse(s);
            for (String c:column){
                csv.map.put(c,new ArrayList<>());
            }
            while ((s = reader.readLine()) != null){
                String[] elements = parse(s);
                for (int i = 0;i < elements.length;++i){
                    csv.map.get(column[i]).add(elements[i]);
                }
            }
            reader.close();
            return csv;
        } catch (IOException e) {
            e.printStackTrace();
            return new CSV();
        }
    }

    public static CSV loadForTradedata(String filePath){
        return loadForTradedata(new File(filePath));
    }

    public static CSV loadForTradedata(File file){
        CSV csv = new CSV();
        if(!file.exists())return csv;
        try {
            BufferedReader reader = new BufferedReader(new FileReader(file));
            reader.readLine();
            String s = reader.readLine();
            if(s == null)return csv;
            String[] column = parse(s);
            for (String c:column){
                csv.map.put(c,new ArrayList<>());
            }
            while ((s = reader.readLine()) != null){
                String[] elements = parse(s);
                for (int i = 0;i < elements.length;++i){
                    csv.map.get(column[i]).add(elements[i]);
                }
            }
            reader.close();
            return csv;
        } catch (IOException e) {
            e.printStackTrace();
            return new CSV();
        }
    }

    private static String[] parse(String in){
        if(in.startsWith("\"")){
            String[] t = in.split("\",\"");
            t[0] = t[0].substring(1);
            if(t.length >= 2){
                String s = t[t.length - 1];
                t[t.length - 1] = s.substring(0,s.length() - 1);
            }
            return t;
        }
        String[] t = in.split(",");
        return t;
    }

}
