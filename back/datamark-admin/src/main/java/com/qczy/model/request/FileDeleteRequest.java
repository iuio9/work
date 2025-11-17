package com.qczy.model.request;

import lombok.Data;

import javax.validation.constraints.NotBlank;
import javax.validation.constraints.NotEmpty;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/27 14:55
 * @Description:
 */
@Data
public class FileDeleteRequest {

    // 数据集id
    @NotBlank(message = "数据集不能为空！")
    private String sonId;

    // 文件id
    @NotEmpty(message = "文件长度必须>0")
    private int[] fileIds;


}
