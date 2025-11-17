package com.qczy.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.qczy.model.entity.DataFatherEntity;
import com.qczy.model.entity.DataSonEntity;
import com.qczy.model.request.DataSetImportRequest;
import com.qczy.model.request.DataSonEntityRequest;
import com.qczy.model.request.UpdateDataSetRequest;

import java.io.IOException;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/12 14:22
 * @Description:
 */
public interface DataFatherService  extends IService<DataFatherEntity>  {

    int deleteDataGroup(String groupId) throws IOException;

    int deleteDataSet(String sonId) throws IOException;

    int dataSetImport( DataSonEntityRequest dataSonRequest);

    int updateDataSetName(UpdateDataSetRequest request);
}
