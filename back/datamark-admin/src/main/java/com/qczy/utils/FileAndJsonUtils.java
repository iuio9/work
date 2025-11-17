package com.qczy.utils;

import com.qczy.mapper.AlgorithmTaskMapper;
import com.qczy.model.entity.AlgorithmTaskEntity;
import com.qczy.model.request.DataSonEntityRequest;
import com.qczy.model.request.ResultDataSonRequest;
import com.qczy.service.DataMarkService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import org.springframework.util.ObjectUtils;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/10/14 15:42
 * @Description:
 */
@Component
public class FileAndJsonUtils {

    @Autowired
    private AlgorithmTaskMapper algorithmTaskMapper;
    @Autowired
    private DataMarkService dataMarkService;



    // 判断任务类型是否为缺陷生成，如果是缺陷生成，则进行图片与json合并
    public void setFlawFileAndJson(ResultDataSonRequest request, DataSonEntityRequest dataSonRequest) {
        AlgorithmTaskEntity taskEntity = algorithmTaskMapper.selectById(request.getTaskId());
        // 执行json跟图片合并
        if (!ObjectUtils.isEmpty(taskEntity)) {
            if (!StringUtils.isEmpty(taskEntity.getModelId()) && taskEntity.getModelId().equals("4")) {
                dataMarkService.setMarkFileJsonWrite(dataSonRequest.getSonId());
            }
        }
    }


}
