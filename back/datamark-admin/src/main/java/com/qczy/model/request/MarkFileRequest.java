package com.qczy.model.request;


import lombok.Data;
import org.springframework.web.multipart.MultipartFile;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/23 19:30
 * @Description:
 */
@Data
public class MarkFileRequest {

    private MultipartFile file;

    private String sonId;

    private Integer version;



}
