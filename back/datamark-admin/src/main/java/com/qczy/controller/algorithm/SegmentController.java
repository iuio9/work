package com.qczy.controller.algorithm;

import cn.hutool.json.JSON;
import cn.hutool.json.JSONUtil;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.common.result.Result;
import com.qczy.model.entity.AlgorithmModelEntity;
import com.qczy.model.entity.AlgorithmTaskEntity;
import com.qczy.model.entity.MarkInfoEntity;
import com.qczy.model.request.DataSonQueryRequest;
import com.qczy.model.request.SegmentParams;
import com.qczy.model.response.DataMarkResponse;
import com.qczy.service.AlgorithmTaskService;
import com.qczy.service.DataMarkService;
import com.qczy.utils.HttpUtil;
import com.qczy.utils.JsonUtil;
import com.qczy.utils.StringUtils;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

/**
 * @Author: gwj
 * @Version: 1.0
 * @Date: 2024/8/22 20:01
 * @Description:
 */
@RestController
@RequestMapping("/algorithm/segment")
@Api(tags = "算法==目标分割")
public class SegmentController {

    @Autowired
    AlgorithmTaskService algorithmTaskService;

    @Autowired
    DataMarkService dataMarkService;


    /**
     *  保存标注信息
     */
    @PostMapping("/start")
    @ApiOperation("算法执行目标分割")
    public Result start(@RequestBody SegmentParams segmentParams){
        Map<String,Object> segment = algorithmTaskService.startSegment(segmentParams);
        return Result.ok(JSONUtil.parseObj(segment));
    }
}
