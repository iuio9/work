package com.qczy.controller.test;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.qczy.common.generate.GenerateWordForm;
import com.qczy.common.result.Result;
import com.qczy.mapper.ModelAssessTaskMapper;
import com.qczy.model.entity.ModelAssessTaskEntity;
import com.qczy.utils.Base64Utils;
import com.qczy.utils.StringUtils;
import lombok.Data;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/5/30 14:14
 * @Description:
 */
@RestController
public class DebugTestController {
    @Autowired
    private GenerateWordForm wordTemplateFiller;
    @Autowired
    private ModelAssessTaskMapper modelAssessTaskMapper;


    private final ObjectMapper objectMapper = new ObjectMapper();

    @GetMapping("/api/test666")
    public String test666() {
      //  wordTemplateFiller.genWord();
        return "hello world";
    }


    @PostMapping("/api/test1")
    public Result test1(@RequestBody Student student) {
        /* return Result.ok("测试成功！！！");*//*
        String name = student.getName();
        Integer age = student.getAge();*/


        if (StringUtils.isEmpty(student.getImage_base64())) {
            return Result.ok("模型调试ok！");
        }


        // 模拟返回信息
        Map<String, Object> data = new HashMap<>();
        // 获取图片的基本信息
        try {
            //String img_path = readTextFile(file);
            Base64Utils.ImageMeta imageMeta = Base64Utils.getImageMeta(student.getImage_base64());
            data.put("img_name", imageMeta.getFileName());
            data.put("img_width", imageMeta.getWidth());
            data.put("img_height", imageMeta.getHeight());


            /**
             *  随机点位，类型
             */
            int count = new Random().nextInt(5) + 1;
            data.put("dec_rsts", generateDetectionResults(count, imageMeta.getWidth(), imageMeta.getHeight()));

            return Result.ok(data);


        } catch (IOException e) {
            throw new RuntimeException(e);
        }


    }


    @GetMapping("/api/test2")
    public Result test2(@RequestParam("name") String name, @RequestParam("age") Integer age) {
        /* return Result.ok("测试成功！！！");*/


        return Result.ok("你传递的参数为： 姓名：" + name + "，年龄：" + age);
    }


    @GetMapping("/api/test3")
    public Result test3() {

        ObjectMapper objectMapper = new ObjectMapper();

        // 创建 HashMap 实例
        Map<String, Object> dataMap = new HashMap<>();
        dataMap.put("PR_curve", "/data/ghf2/Model_Evaluation/evaluation_results/PR_curve.png");
        dataMap.put("confusion_matrix", "/data/ghf2/Model_Evaluation/evaluation_results/confusion_matrix.png");
        dataMap.put("mAP@0.5", 0.12665825251603116);
        dataMap.put("mAccuracy", 0.12278065204894473);
        dataMap.put("mFalseAlarmRate", null);
        dataMap.put("mMissRate", null);
        dataMap.put("mPrecision", 0.19583333333333333);
        dataMap.put("mRecall", 0.17335807467386416);

        try {
            ModelAssessTaskEntity modelAssessTaskEntity = modelAssessTaskMapper.selectById(19);
            modelAssessTaskEntity.setTaskResult(objectMapper.writeValueAsString(dataMap));
            modelAssessTaskMapper.updateById(modelAssessTaskEntity);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }


        return Result.ok("ok!");
    }




//---------------------------------------------------------------------------------------------------------------------------------------------------------

    private static final Random RANDOM = new Random();
    private static final List<String> CLASS_NAMES = Arrays.asList(
            "00001001", "91002002", "00001011"
    );


    // 模拟第三方模型评估供应商接口
    @PostMapping("/api/thirdInvocation")
    //public Result thirdInvocation(@RequestParam(name = "img_base64") String img_base64) {
    public Result thirdInvocation(@RequestBody Base64 base64) {
        if (base64 == null || StringUtils.isEmpty(base64.getImg_base64())) {
            return Result.fail("图片不能为空！");
        }


        /*if (file == null) {
            return Result.fail("imgBase64 不能为空！");
        }
*/

        // 模拟返回信息
        Map<String, Object> data = new HashMap<>();
        // 获取图片的基本信息
        try {
            //String img_path = readTextFile(file);
            Base64Utils.ImageMeta imageMeta = Base64Utils.getImageMeta(base64.getImg_base64());
            data.put("img_name", imageMeta.getFileName());
            data.put("img_width", imageMeta.getWidth());
            data.put("img_height", imageMeta.getHeight());


            /**
             *  随机点位，类型
             */
            //int count = new Random().nextInt(5) + 1;
            data.put("dec_rsts", generateDetectionResults(1, imageMeta.getWidth(), imageMeta.getHeight()));

            return Result.ok(data);


        } catch (IOException e) {
            throw new RuntimeException(e);
        }


    }

    /**
     * 读取文件里面的内容
     *
     * @param file
     * @return
     * @throws IOException
     */
    public String readTextFile(MultipartFile file) throws IOException {
        StringBuilder content = new StringBuilder();

        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(file.getInputStream(), StandardCharsets.UTF_8))) {

            String line;
            while ((line = reader.readLine()) != null) {
                content.append(line);
            }
        }

        return content.toString();
    }

    /**
     * 生成随机检测结果列表
     *
     * @param count       检测结果数量
     * @param imageWidth  图像宽度
     * @param imageHeight 图像高度
     * @return 包含随机边界框、类别和置信度的检测结果
     */
    public static List<Map<String, Object>> generateDetectionResults(int count, int imageWidth, int imageHeight) {
        List<Map<String, Object>> results = new ArrayList<>();
        Random random = new Random();



        for (int i = 0; i < count; i++) {
            int[] bbox = generateRandomBbox(imageWidth, imageHeight);
            String className = CLASS_NAMES.get(random.nextInt(CLASS_NAMES.size()));
            double score = random.nextDouble() * 0.9 + 0.1;

            // 使用 HashMap 替代 Map.of()
            Map<String, Object> result = new HashMap<>();
            result.put("class_name", className);
            result.put("bbox", bbox);
            result.put("score", String.format("%.2f", score));

            results.add(result);
        }

        return results;
    }

    /**
     * 生成随机边界框
     */
    private static int[] generateRandomBbox(int maxWidth, int maxHeight) {
        int x1 = RANDOM.nextInt(maxWidth);
        int y1 = RANDOM.nextInt(maxHeight);
        int x2 = x1 + RANDOM.nextInt(maxWidth - x1);
        int y2 = y1 + RANDOM.nextInt(maxHeight - y1);
        return new int[]{x1, y1, x2, y2};
    }


}

@JsonIgnoreProperties(ignoreUnknown = true) // 忽略未知字段
@Data
class Student {

    private String algCode;
    private String image_base64;


}


@JsonIgnoreProperties(ignoreUnknown = true) // 忽略未知字段
@Data
class Base64 {

    private String img_base64;


}
