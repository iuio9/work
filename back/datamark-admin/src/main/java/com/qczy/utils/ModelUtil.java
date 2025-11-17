package com.qczy.utils;

import cn.hutool.core.collection.ListUtil;
import cn.hutool.core.io.FileUtil;
import cn.hutool.core.map.MapUtil;
import cn.hutool.core.util.NumberUtil;
import cn.hutool.core.util.ObjectUtil;
import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.qczy.controller.dataupload.TempUploadController;
import com.qczy.mapper.AlgorithmMapper;
import com.qczy.model.entity.AlgorithmEntity;
import com.qczy.model.entity.AlgorithmModelEntity;
import com.qczy.model.entity.AlgorithmTaskEntity;
import com.qczy.model.request.AlgorithmParams;

import com.qczy.service.AlgorithmService;
import com.qczy.service.FileService;
import com.qczy.service.impl.AlgorithmServiceImpl;
import io.swagger.models.auth.In;
import org.apache.commons.io.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.multipart.MultipartFile;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @author ：gwj
 * @date ：Created in 2024-08-27 16:40
 * @description：根据算法模型请求Http地址
 * @modified By：
 * @version: $
 */
@Component
public class ModelUtil {
    private static final Logger log = LoggerFactory.getLogger(ModelUtil.class);


    private final HttpUtil httpUtil;
    @Autowired
    private AlgorithmService algorithmService;

    public ModelUtil(HttpUtil httpUtil) {
        this.httpUtil = httpUtil;
    }

    @Value("${upload.formalPath}")
    private String formalPath;
    @Autowired
    FileService fileService;
    public String model(AlgorithmEntity algorithmEntity, AlgorithmParams alParams) {
        // Check if the current algorithm entity is null
        String url = algorithmEntity.getUrl();
        String reqType = algorithmEntity.getRequestType();
        List<Map> paramsMaps = JSONUtil.toList(JSONUtil.parseArray(algorithmEntity.getParams()), Map.class);

        Map<String,Object> algorithmParams = new HashMap<>();
        for (Map paramsMap : paramsMaps) {
            String key = paramsMap.get("serverKey").toString();
            if (paramsMap.get("type").equals("image")) {
                algorithmParams.put(key,URLUtils.encodeURL(alParams.getImage_path()));
            }else if (paramsMap.get("type").equals("json")){
                String httpPath = null;
                String markInfo = alParams.getMarkInfo();
                if(markInfo != null){
                    File file = new File(alParams.getImageAbsoute());
                    String parent = file.getParent();
                    // 查找最后一个点的位置
                    String fileName = file.getName();
                    int dotIndex = fileName.lastIndexOf('.');

                    // 如果存在点，去掉从最后一个点开始的部分

                    if (dotIndex > 0) {
                        fileName = fileName.substring(0, dotIndex);
                    }
                    try {
                        String localDir = parent + "/" + fileName + ".json";
                        File file1 = new File(localDir);
                        FileWriter fileWriter = new FileWriter(file1);
                        fileWriter.write(markInfo);
                        fileWriter.close();
                        System.out.println("Successfully wrote to the file.");
                        httpPath = fileService.addAlgorithmFile(file1,parent);

                        algorithmParams.put(key,URLUtils.encodeURL(httpPath));
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }else if (paramsMap.get("type").equals("text") || paramsMap.get("type").equals("select")){
                String text = alParams.getParams().get(key).toString();


                if(NumberUtil.isNumber(text)){
                    Number number = NumberUtil.parseNumber(text);
                    algorithmParams.put(key, number);
                }else{
                    algorithmParams.put(key, text);
                }

            }
        }



        if(reqType.equalsIgnoreCase("post")){

            System.out.println("发送参数======================"+JSONUtil.toJsonStr(algorithmParams));
            log.info("发送参数======================{}",JSONUtil.toJsonStr(algorithmParams));
            String post = httpUtil.post(url,algorithmParams );
            return post;
        }else if(reqType.equalsIgnoreCase("get")){
            String get = httpUtil.get(url);
            return get;
        }
        return null;
    }
}
