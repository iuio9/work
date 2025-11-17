package com.qczy.common.label;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class JsonParser {
    public static Map<String, String> parseMapLabelToEnglish(String jsonStr) {
        ObjectMapper mapper = new ObjectMapper();
        try {
            List<Map<String, Object>> labelList = mapper.readValue(
                    jsonStr,
                    new TypeReference<List<Map<String, Object>>>() {}
            );

            Map<String, String> resultMap = new HashMap<>();
            for (Map<String, Object> label : labelList) {
                String mapLabel = (String) label.get("mapLabel");
                String englishName = (String) label.get("englishLabelName");
                if (mapLabel != null && englishName != null) {
                    resultMap.put(mapLabel, englishName);
                }
            }
            return resultMap;
        } catch (Exception e) {
            throw new IllegalArgumentException("Invalid JSON format", e);
        }
    }
}