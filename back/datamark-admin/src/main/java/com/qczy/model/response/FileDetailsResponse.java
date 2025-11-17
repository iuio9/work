package com.qczy.model.response;

import lombok.Data;
import org.jpedal.parser.shape.S;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/10/11 9:44
 * @Description:
 */
@Data
public class FileDetailsResponse {

    private Integer id;
    private String fdName;
    private String fdSize;
    private String status;
    private String errorLog;

}
