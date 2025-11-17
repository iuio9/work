package com.qczy.service;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.model.request.DataSonQueryRequest;
import com.qczy.model.response.DataDetailsResponse;
import com.qczy.model.response.DataResponse;
import org.jpedal.parser.shape.S;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/12/24 10:49
 * @Description:
 */
public interface DataSonDetailsService {

    // 查询数据集列表
    IPage<DataResponse> getDataSetList(Page<DataResponse> pageParam, DataSonQueryRequest request);

    // 查询数据集列表不分页
    List<DataResponse> getDataSetListNoPage();

    // 查询当前数据集的详情信息
    IPage<DataDetailsResponse> getDataDetails(Page<DataDetailsResponse> pageParam, String sonId, Integer state, Integer labelId, Integer markUserId, Integer taskId, String sign);

    // 查看原始图
    IPage<DataDetailsResponse> getDataDetailsNoMarkFilePath(Page<DataDetailsResponse> pageParam, String sonId, Integer state,Integer markUserId,Integer taskId,String sign);


    //------------------------------ 查看（全部、有标注信息、五标注信息）分页总数量 ------------------------------
    int selectFileAndlabelCount(String sonId,Integer markUserId,Integer taskId,String sign);

    int selectFileAndlabelYesMarkCount(String sonId,Integer markUserId,Integer taskId,String sign);

    int selectFileAndlabelNoMarkCount(String sonId,Integer markUserId,Integer taskId,String sign);

    int selectFileInvalidDataCount(String sonId,Integer markUserId,Integer taskId,String sign);


    //----------------------------------------------------------------------------------------------------

}
