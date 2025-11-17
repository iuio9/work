package com.qczy.model.entity;

/**
 * @author ：gwj
 * @date ：Created in 2024-11-15 14:17
 * @description：
 * @modified By：
 * @version: $
 */
import cn.hutool.core.bean.BeanUtil;
//import com.qczy.common.annotation.ValidBatchSizeDatabase;
import com.qczy.common.annotation.ValidBatchSizeDatabase;
import lombok.Data;

import javax.validation.constraints.*;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Data
public class TrainingParams {
    @NotNull(message = "epochs不能为空")
    @Pattern(regexp = "100|200|300|400|500", message = "epochs建议是100,200,300,400,500之一")
    private String epochs = "300";


    @NotNull(message = "Weights不能为空")
    @Pattern(regexp = "s|m|l|x", message = "Weights必须是s, m, l, x之一")
    private String weights = "s";

    @NotNull(message = "Batch_size不能为空")
    @Pattern(regexp = "4|8|16|32|64|128", message = "Batch size必须是4, 8, 16, 32, 64, 128之一")
    private String batchSize = "32";

    @NotNull(message = "img_size不能为空")
    @Pattern(regexp = "320|416|640|1024", message = "Image size必须是320, 416, 640, 1024之一")
    private String imgSize;

    @NotNull(message = "image_weights不能为空")
    @Pattern(regexp = "true|false", message = "image_weights只能是true或false")
    private String imageWeights = "false";  // 默认值为 "false"



    @NotNull(message = "name不能为空")
    @Pattern(regexp = "^[a-zA-Z0-9]+$", message = "name只能包含英文字母和数字")
    @ValidBatchSizeDatabase
    private String name;

    @NotNull(message = "epochs_resnet不能为空")
    @Min(value = 30, message = "epochs_resnet至少为30")
    @Max(value = 100, message = "epochs_resnet不能大于100")
    private Integer epochsResnet = 100;

    @NotNull(message = "batch_size_resnet不能为空")
    @Pattern(regexp = "9|16|32|64", message = "ResNet的Batch size必须是9.16, 32, 64之一")
    private String batchSizeResnet = "32";

    @NotNull(message = "learning_rate不能为空")
    @DecimalMin(value = "0.001", message = "learning rate至少为0.001")
    @DecimalMax(value = "0.1", message = "learning rate不能大于0.1")
    private Double learningRate;

    public static void main(String[] args) {
        Map kvHashMap = new HashMap<>();
        kvHashMap.put("batch_size_resnet",88);
        kvHashMap.put("name","gdp");
        TrainingParams trainingParams = BeanUtil.toBean(kvHashMap, TrainingParams.class);
        System.out.println(trainingParams);
    }
}

