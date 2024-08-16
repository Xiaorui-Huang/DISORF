POSE=$1

bash scripts/tnt/splatfacto_eval.sh Barn 532  ${POSE} _uc uncontract
bash scripts/tnt/splatfacto_eval.sh Train 484 ${POSE} _uc uncontract
bash scripts/tnt/splatfacto_eval.sh Truck 314 ${POSE} _uc uncontract

for sampl in default imap inverse_delta hybrid; do
    bash scripts/tnt/splatfacto_eval_online.sh Barn 532 ${POSE} $sampl _uc uncontract 
    bash scripts/tnt/splatfacto_eval_online.sh Train 484 ${POSE} $sampl _uc uncontract
    bash scripts/tnt/splatfacto_eval_online.sh Truck 314 ${POSE} $sampl _uc uncontract
done

for i in {1..2}; do
    for sampl in inverse_delta hybrid; do
        bash scripts/tnt/splatfacto_eval_online.sh Barn 532 ${POSE} $sampl _uc uncontract 
        bash scripts/tnt/splatfacto_eval_online.sh Train 484 ${POSE} $sampl _uc uncontract 
        bash scripts/tnt/splatfacto_eval_online.sh Truck 314 ${POSE} $sampl _uc uncontract 
    done
done

bash scripts/tnt/splatfacto_eval.sh Barn 532  ${POSE}
bash scripts/tnt/splatfacto_eval.sh Train 484 ${POSE}
bash scripts/tnt/splatfacto_eval.sh Truck 314 ${POSE}

for sampl in default imap inverse_delta hybrid; do
    bash scripts/tnt/splatfacto_eval_online.sh Barn 532 ${POSE} $sampl 
    bash scripts/tnt/splatfacto_eval_online.sh Train 484 ${POSE} $sampl
    bash scripts/tnt/splatfacto_eval_online.sh Truck 314 ${POSE} $sampl
done

for i in {1..2}; do
    for sampl in inverse_delta hybrid; do
        bash scripts/tnt/splatfacto_eval_online.sh Barn 532 ${POSE} $sampl 
        bash scripts/tnt/splatfacto_eval_online.sh Train 484 ${POSE} $sampl 
        bash scripts/tnt/splatfacto_eval_online.sh Truck 314 ${POSE} $sampl 
    done
done