package com.qczy.model.request;

import lombok.Data;
import org.springframework.stereotype.Component;

@Data
public class CheckAndClassify {
    private String taskId;
    private String taskName;
}
