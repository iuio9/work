package com.qczy.service;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.qczy.model.entity.DataSonEntity;
import com.qczy.model.request.*;
import com.qczy.model.response.DataDetailsResponse;
import com.qczy.model.response.DataResponse;
import org.apache.ibatis.annotations.Param;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/7 21:49
 * @Description:
 */
public interface DataSonService extends IService<DataSonEntity> {

    // 新增数据集
    int insertDataSet(DataSonEntityRequest dataSonRequest);

    // 导入
    int getResultDataSetSave(ResultDataSonRequest request);

    // 新增数据集版本
    int addDataVersion(SaveSonVersionRequest saveSonVersionRequest);

    // 修改数据集备注
    int updateDataSetRemark(UpdateDataSetRequest request);



    void setFlawFileAndJson(ResultDataSonRequest request, DataSonEntityRequest dataSonRequest);

}
