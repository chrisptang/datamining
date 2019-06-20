package com.tangpeng.datamining;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

public class PrintUtil {
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    public static void print(Object object) {
        if (null == object) {
            return;
        }
        if (object instanceof String) {
            System.out.println(object);
            return;
        }
        String result;
        try {
            result = OBJECT_MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(object);
            System.out.println(result);
        } catch (JsonProcessingException e) {
            e.printStackTrace();
        }
    }
}
