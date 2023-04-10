const express = require('express');
const router = express.Router();
const googleTrends = require('google-trends-api');

// router.post('/trends', async (req,res) =>{
//   googleTrends.interestOverTime({keyword: req.body.trendfor})
//   .then(function(results){
//     return res.status(200).send({'trends': JSON.parse(results)});
//   })
//   .catch(function(err){
//     res.status(404).send(err);
//   });

// });

router.post('/trends', async (req, res) => {
  googleTrends.interestOverTime({
    keyword: req.body.trendfor,
    geo: 'PK'
  })
    .then(function(results) {
      return res.status(200).send({ trends: JSON.parse(results) });
    })
    .catch(function(err) {
      res.status(404).send(err);
    });
});


module.exports = router;