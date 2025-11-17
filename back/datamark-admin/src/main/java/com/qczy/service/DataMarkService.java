package com.qczy.service;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.qczy.model.entity.MarkInfoEntity;
import com.qczy.model.request.DataSonQueryRequest;
import com.qczy.model.response.DataDetailsResponse;
import com.qczy.model.response.DataMarkResponse;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/22 20:05
 * @Description:
 */
public interface DataMarkService extends IService<MarkInfoEntity> {

    IPage<DataMarkResponse> getDataSetMarkList(Page<DataMarkResponse> pageParam, DataSonQueryRequest request);

    int addDataMarkInfo(MarkInfoEntity markInfoEntity);

    List<DataDetailsResponse> getDataDetails(String sonId, Integer state);


    // 提供图片json加工数据
    void setMarkFileJsonWrite(String outSonId);

    /**
     *
     * @param sonId  数据集id
     * @param fileId 原始图文件id
     * @param jsonId json文件id
     * @param markFileId mark文件id
     */
    // 自动标注 （根据图片+json+mark图片加工数据）
    void setMarkFileJsonAndMarkFileWrite(String sonId,String fileId,
                                         String jsonId,String markFileId);







}
